#!/usr/bin/env python
import os
import random
import torch
import json
from datasets import load_dataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from gram_model import buildDataset, gramModel, processTestInput
from preprocess_text import preprocess_example, preprocess_text


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    @classmethod
    def __init__(self):
        self.gram2 = []
        self.gram3 = []
        self.gram4 = []
        self.mod2 = gramModel(2, 0)
        self.mod3 = gramModel(3,0)
        self.mod4 = gramModel(4,0)
        self.ctoi = {}
        self.itoc = {}

    @classmethod
    def load_training_data(cls):
        dataset = load_dataset('wikipedia', '20220301.en', split='train[:1%]', trust_remote_code= True)  # 1% of enwik8 for training
        small_dataset = dataset.shuffle(seed=42).select(range(int(0.001 * len(dataset))))

        small_dataset = small_dataset.map(preprocess_example)
        small_text = ' '.join(small_dataset['text'])

        X2_train, Y2_train, ctoi2, itoc2 = buildDataset(small_dataset, small_text, 2)
        X3_train, Y3_train, ctoi3, itoc3 = buildDataset(small_dataset, small_text, 3)
        X4_train, Y4_train, ctoi4, itoc4 = buildDataset(small_dataset, small_text, 4)
        return [[X2_train, Y2_train, ctoi2, itoc2], [X3_train, Y3_train, ctoi3, itoc3], [X4_train, Y4_train, ctoi4, itoc4]]

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                inp = preprocess_text(inp)
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        ## of the form [xTrain, yTrain, ctoi, itoc]
        ## where ctoi - char to index
        ## and itoc - index to char
        model2Data = data[0]
        model3Data = data[1]
        model4Data = data[2]

        ## build models
        model2 = gramModel(block_size = 2, vocab_size = len(model2Data[2]))
        model3 = gramModel(block_size = 3, vocab_size = len(model3Data[2]))
        model4 = gramModel(block_size = 4, vocab_size = len(model4Data[2]))

        ## train models
        model2.train(model2Data[0], model2Data[1])
        print("bigram model done training")

        model3.train(model3Data[0], model3Data[1])
        print("trigram model done training")

        model4.train(model4Data[0], model4Data[1])
        print("quadgram model done training")

        self.gram2 = model2.parameters
        self.gram3 = model3.parameters
        self.gram4 = model4.parameters
        self.ctoi = model2Data[2]
        self.itoc = model2Data[3]

    def run_pred(self, data):
        preds = []
        for inp in data:
            probMod2 = self.mod2.get_prob(processTestInput(self.ctoi, inp, block_size=2))
            probMod3 = self.mod3.get_prob(processTestInput(self.ctoi, inp, block_size=3))
            probMod4 = self.mod4.get_prob(processTestInput(self.ctoi, inp, block_size=4))

            overallProb = torch.mean(torch.stack([probMod2, probMod3, probMod4]), dim=0)
            overallProb = overallProb.view(-1)

            topIndex  = torch.argsort(overallProb, dim = 0, descending=True)[:3].tolist()
            
            top_guesses = [self.itoc[str(index)] for index in topIndex]
            preds.append(''.join(top_guesses))

            
        return preds

    def save(self, work_dir):
        torch.save(self.gram2, os.path.join(work_dir, 'model.checkpoint2'))
        torch.save(self.gram3, os.path.join(work_dir, 'model.checkpoint3'))
        torch.save(self.gram4, os.path.join(work_dir, 'model.checkpoint4'))

        with open(os.path.join(work_dir, 'model.checkpointCTOI'), 'wt') as f:
            json.dump(self.ctoi, f)

        with open(os.path.join(work_dir, 'model.checkpointITOC'), 'wt') as f:
            json.dump(self.itoc, f)



    @classmethod
    def load(cls, work_dir):
        gram2 = torch.load(os.path.join(work_dir, 'model.checkpoint2'))
        gram3 = torch.load(os.path.join(work_dir, 'model.checkpoint3'))
        gram4 = torch.load(os.path.join(work_dir, 'model.checkpoint4'))

        with open(os.path.join(work_dir, 'model.checkpointCTOI')) as f:
            ctoi = json.load(f)

        with open(os.path.join(work_dir, 'model.checkpointITOC')) as f:
            itoc = json.load(f)

        model = MyModel()
        model.gram2 = gram2
        model.gram3 = gram3
        model.gram4 = gram4
        model.ctoi = ctoi
        model.itoc = itoc

        ## create new models
        mod2 = gramModel(block_size=2, vocab_size = len(ctoi))
        mod3 = gramModel(block_size = 3, vocab_size = len(ctoi))
        mod4 = gramModel(block_size = 4, vocab_size= len(ctoi))

        ## set weights of models from saved data
        mod2.embd = gram2[0]
        mod2.W1 = gram2[1]
        mod2.B1 = gram2[2]
        mod2.W2 = gram2[3]
        mod2.B2 = gram2[4]
        mod2.W3 = gram2[5]
        mod2.B3 = gram2[6]

        mod3.embd = gram3[0]
        mod3.W1 = gram3[1]
        mod3.B1 = gram3[2]
        mod3.W2 = gram3[3]
        mod3.B2 = gram3[4]
        mod3.W3 = gram3[5]
        mod3.B3 = gram3[6]

        mod4.embd = gram4[0]
        mod4.W1 = gram4[1]
        mod4.B1 = gram4[2]
        mod4.W2 = gram4[3]
        mod4.B2 = gram4[4]
        mod4.W3 = gram4[5]
        mod4.B3 = gram4[6]

        ## save models to overall model
        model.mod2 = mod2
        model.mod3 = mod3
        model.mod4 = mod4

        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
