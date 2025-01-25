#!/usr/bin/env python
import os
import string
import random
from datasets import load_dataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from gram_model import buildDataset, gramModel, processTestInput
from preprocess_text import preprocess_text, preprocess_example


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

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
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
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

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


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
