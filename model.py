import torch
from torch import nn
from transformers import AutoModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class MyModel(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.cls_id = 0
        hidden_dim = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, args.num_labels)
                )

    def forward(self, x, mask):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor to the model.
            mask (torch.Tensor): Attention mask tensor.

        Returns:
            tuple: A tuple containing:
                - out (transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions): Output from the backbone model.
                - torch.Tensor: Output from the classifier applied to the last hidden state of the backbone model.
        """
        x = x.to(self.backbone.device)
        mask = mask.to(self.backbone.device)
        out = self.backbone(x, attention_mask = mask, output_attentions=True)
        return out, self.classifier(out.last_hidden_state)

    def decisions(self, x, mask):
        """
        Make decisions based on the input data and mask.

        Args:
            x (torch.Tensor): Input tensor to the model.
            mask (torch.Tensor): Attention mask tensor.

        Returns:
            tuple: A tuple containing the output from the backbone model and the classifier output.
        """
        x = x.to(self.backbone.device)
        mask = mask.to(self.backbone.device)
        out = self.backbone(x, attention_mask = mask, output_attentions=False)
        return out, self.classifier(out.last_hidden_state)

    def phenos(self, x, mask):
        """
        Processes the input tensor `x` and its corresponding `mask` through the model's backbone and classifier.

        Args:
            x (torch.Tensor): The input tensor to be processed by the model.
            mask (torch.Tensor): The attention mask tensor corresponding to the input tensor `x`.

        Returns:
            tuple: A tuple containing:
                - out (transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions): The output from the backbone model, including attentions.
                - torch.Tensor: The output from the classifier applied to the pooler output of the backbone model.
        """
        x = x.to(self.backbone.device)
        mask = mask.to(self.backbone.device)
        out = self.backbone(x, attention_mask = mask, output_attentions=True)
        return out, self.classifier(out.pooler_output)

    def generate(self, x, mask, choice=None):
        """
        Generates the output based on the input sequence and mask.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        mask (torch.Tensor): Mask tensor of shape (batch_size, sequence_length).
        choice (str, optional): Specifies the task type. Can be 'seq' or 'token'. Defaults to None.

        Returns:
        torch.Tensor: The generated output tensor.
        """
        outs = []
        if self.args.task == 'seq' or choice == 'seq':
            for i, offset in enumerate(range(0, x.shape[1], self.args.max_len-1)):
                if i == 0:
                    segment = x[:, offset:offset + self.args.max_len-1]
                    segment_mask = mask[:, offset:offset + self.args.max_len-1]
                else:
                    segment = torch.cat((torch.ones((x.shape[0], 1), dtype=int).to(x.device)\
                            *self.cls_id,
                            x[:, offset:offset + self.args.max_len-1]), axis=1)
                    segment_mask = torch.cat((torch.ones((mask.shape[0], 1)).to(mask.device),
                            mask[:, offset:offset + self.args.max_len-1]), axis=1)
                logits = self.phenos(segment, segment_mask)[1]
                outs.append(logits)

            return torch.max(torch.stack(outs, 1), 1).values
        elif self.args.task == 'token':
            for i, offset in enumerate(range(0, x.shape[1], self.args.max_len)):
                segment = x[:, offset:offset + self.args.max_len]
                segment_mask = mask[:, offset:offset + self.args.max_len]
                h = self.decisions(segment, segment_mask)[0].last_hidden_state
                outs.append(h)
            h = torch.cat(outs, 1)
            return self.classifier(h)


def load_model(args, device):
    """
    Load and initialize the model, criterion, optimizer, and learning rate scheduler.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - model (str): The type of model to load ('lstm', 'cnn', or a custom model name).
            - model_name (str): The name of the pretrained model to load (if applicable).
            - ckpt (str): Path to the checkpoint file to load the model state from (if applicable).
            - label_encoding (str): The type of label encoding ('multiclass' or other).
            - use_crf (bool): Whether to use Conditional Random Field (CRF) for sequence labeling.
            - num_labels (int): The number of labels for the classification task.
            - pos_weight (float): The positive class weight for binary classification.
            - lr (float): The learning rate for the optimizer.
            - total_steps (int): The total number of training steps for the learning rate scheduler.
        device (torch.device): The device to load the model onto (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the following elements:
            - model (torch.nn.Module): The initialized model.
            - crit (torch.nn.Module): The criterion (loss function) for training.
            - optimizer (torch.optim.Optimizer): The optimizer for training.
            - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
    """
    if args.model == 'lstm':
        model = LSTM(args).to(device)
        model.device = device
    elif args.model == 'cnn':
        model = CNN(args).to(device)
        model.device = device
    else:
        model = MyModel(args, AutoModel.from_pretrained(args.model_name)).to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
    if args.label_encoding == 'multiclass':
        if args.use_crf:
            from torchcrf import CRF
            crit = CRF(args.num_labels, batch_first = True).to(device)
        else:
            crit = nn.CrossEntropyLoss(reduction='none')
    else:
        crit = nn.BCEWithLogitsLoss(
                pos_weight=torch.ones(args.num_labels).to(device)*args.pos_weight,
                reduction='none'
                )
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
            int(0.1*args.total_steps), args.total_steps)

    return model, crit, optimizer, lr_scheduler
