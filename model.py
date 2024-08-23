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
        x = x.to(self.backbone.device)
        mask = mask.to(self.backbone.device)
        out = self.backbone(x, attention_mask = mask, output_attentions=True)
        return out, self.classifier(out.last_hidden_state)

    def decisions(self, x, mask):
        x = x.to(self.backbone.device)
        mask = mask.to(self.backbone.device)
        out = self.backbone(x, attention_mask = mask, output_attentions=False)
        return out, self.classifier(out.last_hidden_state)

    def phenos(self, x, mask):
        x = x.to(self.backbone.device)
        mask = mask.to(self.backbone.device)
        out = self.backbone(x, attention_mask = mask, output_attentions=True)
        return out, self.classifier(out.pooler_output)

    def generate(self, x, mask, choice=None):
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
