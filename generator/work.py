import torch
import sacrebleu
import json

from torch.utils.data import DataLoader

from data import Vocab, MyDataset, STR, END, CLS, rCLS, SEL, sparsify_batch, worker_init_fn
from generator import Generator
from extract import LexicalMap, read_file
from utils import move_to_cuda
from postprocess import PostProcess
import argparse, os

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--test_forests', type=str)
    parser.add_argument('--test_sep2frags', type=str)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--beam_size', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--max_time_step', type=int)
    parser.add_argument('--output_suffix', type=str)
    parser.add_argument('--device', type=int, default=0)

    return parser.parse_args()

def generate_batch(model, batch, beam_size, alpha, max_time_step):
    sparsify_batch(batch)
    batch = move_to_cuda(batch, model.device)
    res = dict()
    token_batch, score_batch = [], []
    beams = model.work(batch, beam_size, max_time_step)
    for beam in beams:
        best_hyp = beam.get_k_best(1, alpha)[0]
        predicted_token = [token for token in best_hyp.seq[1:-1]]
        token_batch.append(predicted_token)
        score_batch.append(best_hyp.score)
    res['token'] = token_batch
    res['score'] = score_batch
    return res

def validate(model, test_data, golden_file, beam_size=8, alpha=0.6, max_time_step=100):
    """For development only"""
    pp = PostProcess()

    ref_stream = []
    for line in open(golden_file+'.input_clean'):
        if line.startswith('# ::tokens '):
            o = json.loads(line[len('# ::tokens '):].strip())
            ref_stream.append(' '.join(o).lower())
    # gold model output
    graph, gold_sys_stream, _, abstract = read_file(golden_file+'.preproc')
    ref_streams = [ref_stream]

    sys_stream = []
    for batch in test_data:
        res = generate_batch(model, batch, beam_size, alpha, max_time_step)
        sys_stream.extend(res['token'])

    assert len(sys_stream) == len(ref_stream)
    sys_stream = [ pp.post_process(o, abstract[i], graph[i]) for i, o in enumerate(sys_stream)]

    bleu = sacrebleu.corpus_bleu(sys_stream, ref_streams,
                          force=True, lowercase=True,
                          tokenize='none').score
    chrf = sacrebleu.corpus_chrf(sys_stream, ref_streams)

    return bleu, chrf

if __name__ == "__main__":

    args = parse_config()

    test_models = []
    if os.path.isdir(args.load_path):
        for file in os.listdir(args.load_path):
            fname = os.path.join(args.load_path, file)
            if os.path.isfile(fname):
                test_models.append(fname)
        model_args = torch.load(fname)['args']
    else:
        test_models.append(args.load_path)
        model_args = torch.load(args.load_path)['args']
    vocabs = dict()
    vocabs['concept'] = Vocab(model_args.concept_vocab, 5, [CLS])
    vocabs['token'] = Vocab(model_args.token_vocab, 5, [STR, END])
    vocabs['predictable_token'] = Vocab(model_args.predictable_token_vocab, 5, [END])
    vocabs['token_char'] = Vocab(model_args.token_char_vocab, 100, [STR, END])
    vocabs['concept_char'] = Vocab(model_args.concept_char_vocab, 100, [STR, END])
    vocabs['relation'] = Vocab(model_args.relation_vocab, 5, [CLS, rCLS, SEL])
    lexical_mapping = LexicalMap()

    if args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = Generator(vocabs,
            model_args.token_char_dim, model_args.token_dim,
            model_args.concept_char_dim, model_args.concept_dim,
            model_args.cnn_filters, model_args.char2word_dim, model_args.char2concept_dim,
            model_args.rel_dim, model_args.rnn_hidden_size, model_args.rnn_num_layers,
            model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout,
            model_args.snt_layers, model_args.graph_layers,
            model_args.inference_layers, model_args.pretrained_file, device)

    test_data = MyDataset(vocabs, lexical_mapping, args.test_data, args.test_forests, args.test_sep2frags, args.test_batch_size, for_train=False)
    test_loader = DataLoader(test_data, batch_size=None, worker_init_fn=worker_init_fn, num_workers=1)
    for test_model in test_models:
        print (test_model)
        model.load_state_dict(torch.load(test_model)['model_state_dict'])
        model = model.cuda(device)
        model.eval()

        tot = 0
        with open(test_model+args.output_suffix, 'w') as fo:
            for batch in test_loader:
                res = generate_batch(model, batch, args.beam_size, args.alpha, args.max_time_step)
                for token in res['token']:
                    fo.write('#model output:' + ' '.join(token) + '\n')
                    tot += 1
        print ('write down %d sentences'%tot)
