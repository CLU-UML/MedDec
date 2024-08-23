import copy
import torch
from torch import nn
from transformers import AutoModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
# from torchcrf import CRF

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

class CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(args.vocab_size, args.emb_size)
        self.model = nn.Sequential(
                nn.Conv1d(args.emb_size, args.hidden_size, args.kernels[0],
                    padding='same' if args.task == 'token' else 'valid'),
                nn.ReLU(),
                nn.MaxPool1d(1),
                nn.Conv1d(args.hidden_size, args.hidden_size, args.kernels[1],
                    padding='same' if args.task == 'token' else 'valid'),
                nn.ReLU(),
                nn.MaxPool1d(1),
                nn.Conv1d(args.hidden_size, args.hidden_size, args.kernels[2],
                    padding='same' if args.task == 'token' else 'valid'),
                nn.ReLU(),
                nn.MaxPool1d(1),
                )
        if args.task == 'seq':
            out_shape = 512 - args.kernels[0] - args.kernels[1] - args.kernels[2] + 3
        elif args.task == 'token':
            out_shape = 1
        self.classifier = nn.Linear(args.hidden_size*out_shape, args.num_labels)
        self.dropout = nn.Dropout()
        self.args = args
        self.device = None

    def forward(self, x, _):
        x = x.to(self.device)
        bs = x.shape[0]
        x = self.emb(x)
        x = x.transpose(1,2)
        x = self.model(x)
        x = self.dropout(x)
        if self.args.task == 'token':
            x = x.transpose(1,2)
            h = self.classifier(x)
            return x, h
        elif self.args.task == 'seq':
            x = x.reshape(bs, -1)
            x = self.classifier(x)
            return x

    def generate(self, x, _):
        outs = []
        for i, offset in enumerate(range(0, x.shape[1], self.args.max_len)):
            segment = x[:, offset:offset + self.args.max_len]
            n = segment.shape[1]
            if n != self.args.max_len:
                segment = torch.nn.functional.pad(segment, (0, self.args.max_len -  n))
            if self.args.task == 'seq':
                logits = self(segment, None)
                outs.append(logits)
            elif self.args.task == 'token':
                h = self(segment, None)[0]
                h = h[:,:n]
                outs.append(h)
        if self.args.task == 'seq':
            return torch.max(torch.stack(outs, 1), 1).values
        elif self.args.task == 'token':
            h = torch.cat(outs, 1)
            return self.classifier(h)

class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(args.vocab_size, args.emb_size)
        self.model = nn.LSTM(args.emb_size, args.hidden_size, num_layers=args.num_layers,
                batch_first=True, bidirectional=True)
        dim = 2*args.num_layers*args.hidden_size if args.task == 'seq' else 2*args.hidden_size
        self.classifier = nn.Linear(dim, args.num_labels)
        self.dropout = nn.Dropout()
        self.args = args
        self.device = None

    def forward(self, x, _):
        x = x.to(self.device)
        x = self.emb(x)
        o, (x, _) = self.model(x)
        o_out = self.classifier(o) if self.args.task == 'token' else None
        if self.args.task == 'seq':
            x = torch.cat([h for h in x], 1)
            x = self.dropout(x)
            x = self.classifier(x)
        return (x, o), o_out

    def generate(self, x, _):
        outs = []
        for i, offset in enumerate(range(0, x.shape[1], self.args.max_len)):
            segment = x[:, offset:offset + self.args.max_len]
            if self.args.task == 'seq':
                logits = self(segment, None)[0][0]
                outs.append(logits)
            elif self.args.task == 'token':
                h = self(segment, None)[0][1]
                outs.append(h)
        if self.args.task == 'seq':
            return torch.max(torch.stack(outs, 1), 1).values
        elif self.args.task == 'token':
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
