import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN,LSTM, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras import metrics
from hla_support import prepare_data_set
from metrics_binarized import *
from tensorflow.keras.utils import plot_model
import numpy as np
import argparse
import pickle

def build_model(input_dim, 
                output_dim, 
                embedding_dim, 
                hidden_dim, 
                dense_dim, 
                activation, 
                input_length,
                lrate=0.001
            ):
    '''
    Build a model with the specified hyperparameters
    '''
    # Create the model
    model = Sequential()
    # Add an Embedding layer
    model.add(Embedding(input_dim, embedding_dim, input_length=input_length))
    # Add a RNN layer -- GRU in this case
    model.add(GRU(hidden_dim, return_sequences=False,unroll=True))
    #Iterating over dense_dim list to add dense layers
    for d in dense_dim:
        model.add(Dense(d, activation=activation))

    # Add a Dense layer with sigmoid activation
    model.add(Dense(output_dim, activation='sigmoid'))
    # Compile the model
    opt = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon= None, decay=0.0, amsgrad= False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.BinaryAccuracy(),metrics.AUC()])
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

def train_model(model,
                ins_train,
                outs_train,
                ins_valid,
                outs_valid,
                ins_test,
                outs_test,
                args
                ):

    '''
    Train the model
    '''
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                restore_best_weights=True,
                                                min_delta=args.min_delta
                                                )

    history = model.fit(ins_train, 
                        outs_train, 
                        epochs=args.epochs, 
                        batch_size=args.batch_size, 
                        validation_data=(ins_valid, outs_valid),
                        verbose=args.verbose, 
                        callbacks=[early_stopping_cb])
            
    # Generate results data
    results = {}
    results['args'] = args
    results['predict_validation'] = model.predict(ins_valid)
    results['predict_validation_eval'] = model.evaluate(ins_valid, outs_valid, verbose=0)
    
    results['predict_testing'] = model.predict(ins_test)
    results['predict_testing_eval'] = model.evaluate(ins_test, outs_test, verbose=0)
        
    results['predict_training'] = model.predict(ins_train)
    results['predict_training_eval'] = model.evaluate(ins_train, outs_train, verbose=0)
    results['history'] = history.history

    # TODO: really want to know the order of the objects in the validation and testing dataset
    
    with open("results_fold_%d.pkl"%fold, "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    model.save("model_%d"%(args.fold))

def create_parser():
    '''
    Create argument parser

    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='RNN', fromfile_prefix_chars='@')

    # Required parameters
    parser.add_argument('--fold', default=None, type=int, required=True,help='fold number')
    parser.add_argument('--verbose', '-v', action = 'count', required=False, help='verbosity level')
    parser.add_argument('--epochs',  default=10, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--embedding_dim', default=32, type=int, help='embedding dimension')
    parser.add_argument('--hidden_dim',  default=32, type=int, help='hidden dimension')
    parser.add_argument('--learning_rate', default=0.001, type=float,  help='learning rate')
    parser.add_argument('--dense_dim',  nargs='+', default=[20,10], type=int, help='output dimension')
    parser.add_argument('--output_dim',   default=1, type=int, help='output dimension')
    parser.add_argument('--validation_fraction', type=float, default=0.1, help='validation fraction')
    parser.add_argument('--results_path', type=str, default='/home/fagg/datasets/HLA', help='path to results')
    parser.add_argument('--activation', type=str, default='relu', help='activation function')
    parser.add_argument('--patience', type=int, default=10, help='patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='min delta')

    return parser

if __name__ == '__main__':
    '''
    Main function
    '''
    parser = create_parser()
    args = parser.parse_args()
    
    if args.fold is not None:
        
        # Load the data set
        tokenizer, len_max, n_tokens, ins_train, outs_train,\
        ins_valid, outs_valid, ins_test, outs_test = prepare_data_set(fold=args.fold, 
                                                                        dir_base=args.results_path,
                                                                        allele = '1301', seed = 100, 
                                                                        valid_size = args.validation_fraction
                                                                    )

        model = build_model(input_dim=n_tokens, 
                            output_dim=args.output_dim,
                            dense_dim=args.dense_dim, 
                            embedding_dim=args.embedding_dim, 
                            hidden_dim=args.hidden_dim, 
                            input_length=len_max,
                            lrate=args.learning_rate,
                            activation=args.activation)

        train_model(model,ins_train,outs_train,ins_valid,outs_valid,ins_test,outs_test,args)
      




