from builtins import breakpoint, print
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from models.models import EHFBClassifier
from models.layer import get_embedding
from transformers import BertModel, AdamW
from data_utils import build_senticNet, Tokenizer4BertGCN, ABSAGCNData, ABSA_collate_fn
from prepare_vocab import VocabHelp
from tensorboardX import SummaryWriter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
writer = SummaryWriter('curvs')

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        
        # load knowledge graph
        bert_vocab={}
        with open(r'vocab/vocab.txt','r', encoding='utf-8') as f:
            for num,line in enumerate(f.readlines()):
                bert_vocab[line.strip()]=num
        graph_embeddings = get_embedding(bert_vocab, opt)
        
        tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')
        self.model = opt.model_class(bert, graph_embeddings, opt).to(opt.device)
        trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, opt, dep_vocab)
        testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt, dep_vocab)
            
        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True, drop_last=True, collate_fn=ABSA_collate_fn)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size, drop_last=True, collate_fn=ABSA_collate_fn)

        self._print_args()
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')
        
        for arg in vars(self.opt):
              logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def get_bert_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        logger.info("bert learning rate on")
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.opt.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        device = self.opt.device
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                self.model.train()
                optimizer.zero_grad()

                batch = [b.to(device) for b in sample_batched]
                inputs = batch[:-1]     # 这是涉及到一个切片的操作，所以直接去掉最后一个变量
                targets = batch[-1]

                outputs, multi_loss = self.model(inputs)          
                Lc = criterion(outputs.view(-1, 3), targets.view(-1))
                Ldep = multi_loss
                loss = self.opt.gamma * Lc + self.opt.theta * Ldep

                loss.backward()
                optimizer.step()
                
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
            
        return max_test_acc, max_f1, model_path
    
    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        # self.model.load_state_dict(torch.load('state_dict/EHFB_restaurant_acc_0.8814_f1_0.8278'))
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        device = self.opt.device

        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                batch = [b.to(device) for b in sample_batched]
                inputs = batch[:-1]     # 这是涉及到一个切片的操作，所以直接去掉最后一个变量
                targets = batch[-1]
                outputs, _ = self.model(inputs)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1
        
    
    def run(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
       
        logger.info('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))


def main():
    model_classes = {
        'EHFB': EHFBClassifier,
    }
    
    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train_new.json',
            'test': './dataset/Restaurants_corenlp/test_new.json',
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train_new.json',
            'test': './dataset/Laptops_corenlp/test_new.json'
        },
        'twitter': {
            'train': './dataset/Tweets_corenlp/train_new.json',
            'test': './dataset/Tweets_corenlp/test_new.json',
        }
    }
    
    # input_colses = {
    #     'latgatsembert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'src_mask', 'aspect_mask', 'lex', 'ori_tag', 'head', 'con_spans'],
    # }
    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad, 
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax, 
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='EHFB', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='restaurant', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=15, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')   
    parser.add_argument('--attention_heads', default=1, type=int, help='number of multi-attention heads')
    parser.add_argument('--max_length', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--vocab_dir', type=str, default='./dataset/Restaurants_corenlp')
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--cuda', default='0', type=str)

    # hyper-parameter
    parser.add_argument('--gamma', default=1, type=float, help="The balance of main task loss.")
    parser.add_argument('--theta', default=0, type=float, help="The balance of root loss.")
    parser.add_argument('--alpha', default=0, type=float, help="The balance of root loss.")
    
    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--bert_lr', default=2e-5, type=float)

    # dep, con, seman and knowledge
    parser.add_argument('--dep_size', default=384, type=int)
    parser.add_argument('--lower', default=True, help = 'lowercase all words.')
    parser.add_argument('--special_token', default='[N]')
    parser.add_argument('--max_num_spans', type=int, default=3, help='inner encoder layers')
    parser.add_argument('--syn_condition', default='con_and_dep', type=str, help='con_dot_dep, con_and_dep')
    parser.add_argument('--dim_w', type=int, default=768, help="dimension of word embeddings")
    parser.add_argument('--lstm_dim', type=int, default=384, help="dimension of bi-lstm")
    parser.add_argument('--is_bert', type=int, default=1, help="glove-based model: 1 for bert")
    parser.add_argument('--dim_k', type=int, default=400, help="dimension of knowledge graph embeddings, 400 for laptop, 200 for rest")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="dropout rate for sentimental features")
    parser.add_argument('--fusion_condition', default='ResEMFH', type=str, help='[ConvIteract, Triaffine, ResEMFH]')
    parser.add_argument('--dep_layers', type=int, default=0)
    parser.add_argument('--sem_layers', type=int, default=0)

    # Interact(EMFN)
    parser.add_argument('--high_order', type=bool, default=True)
    parser.add_argument('--hidden_size', type=int, default=512, help='lower dimension, 512, 400')
    parser.add_argument('--dropout_r', type=float, default=0.1)
    parser.add_argument('--n_layers', type=int, default=2)

    opt = parser.parse_args()
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    # opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

    setup_seed(opt.seed)

    if not os.path.exists('./log'):
        os.makedirs('./log', mode=0o777)
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H_%M_%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

    ins = Instructor(opt)
    ins.run()
    writer.close()

if __name__ == '__main__':
    main()
