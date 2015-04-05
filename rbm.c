//Simple RBM Implementation

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "rbm.h"

#define NUM_HIDDEN 2
#define NUM_VISIBLE 6
#define TRAIN_SIZE 6
#define NUM_EPOCHS 5000
#define LEARN_RATE 0.1

//#define DEBUG

//Weights Vector with Bias Units
//in first col/row
double weights[NUM_VISIBLE+1][NUM_HIDDEN+1];

double pos_hidden_activations[TRAIN_SIZE][NUM_HIDDEN+1];
double pos_hidden_probs[TRAIN_SIZE][NUM_HIDDEN+1];
double pos_assoc[NUM_VISIBLE+1][NUM_HIDDEN+1];

double neg_visible_activations[TRAIN_SIZE][NUM_VISIBLE+1];
double neg_visible_probs[TRAIN_SIZE][NUM_VISIBLE+1];

double neg_hidden_activations[TRAIN_SIZE][NUM_HIDDEN+1];
double neg_hidden_probs[TRAIN_SIZE][NUM_HIDDEN+1];

double neg_assoc[NUM_VISIBLE+1][NUM_HIDDEN+1];

double logistic(double x){
    return (1/(1+exp(-x)));
}

void initialize_weights(){
    int i,j;
    for(i=0;i<NUM_VISIBLE+1;i++){
        for(j=0; j<NUM_HIDDEN+1;j++){
            //weights[i][j] = 0;
            weights[i][j] = 0.1*rand_twister();
            if(rand_twister()>0.5){
                weights[i][j] = -weights[i][j];
            }
            if(i==0 || j==0){
                weights[i][j] = 0;
            }
        }
    }

#ifdef DEBUG
    printf("Initial Weights\n");
    for(i=0; i<NUM_VISIBLE+1; i++){
        for(j=0; j<NUM_HIDDEN+1; j++){
            printf("%f ", weights[i][j]);
        }
        printf("\n");
    }
#endif 

}

void print_weights(){
    int i, j;
    printf("\nCurrent Weights\n");
    for(i=0; i<NUM_VISIBLE+1; i++){
        for(j=0; j<NUM_HIDDEN+1; j++){
            printf("%f ", weights[i][j]);
        }
        printf("\n");
    }
}

void zero_arrays(){
/* Must Zero Out These Arrays
double pos_hidden_activations[TRAIN_SIZE][NUM_HIDDEN+1];
double pos_hidden_probs[TRAIN_SIZE][NUM_HIDDEN+1];
double pos_assoc[NUM_VISIBLE+1][NUM_HIDDEN+1];

double neg_visible_activations[TRAIN_SIZE][NUM_VISIBLE+1];
double neg_visible_probs[TRAIN_SIZE][NUM_VISIBLE+1];

double neg_hidden_activations[TRAIN_SIZE][NUM_HIDDEN+1];
double neg_hidden_probs[TRAIN_SIZE][NUM_HIDDEN+1];

double neg_assoc[NUM_VISIBLE+1][NUM_HIDDEN+1];
*/
    int i, j;
    //Zero Out Pos Arrays
    for(i=0;i<TRAIN_SIZE;i++){
        for(j=0;j<NUM_HIDDEN+1;j++){
            pos_hidden_activations[i][j] = 0;
            pos_hidden_probs[i][j] = 0;
            neg_hidden_activations[i][j] = 0;
            neg_hidden_probs[i][j] = 0;
        }
    }
    for(i=0; i<NUM_VISIBLE+1; i++){
        for(j=0; j<NUM_HIDDEN+1; j++){
            pos_assoc[i][j] = 0;
            neg_assoc[i][j] = 0;
        }
    }
    for(i=0; i<TRAIN_SIZE; i++){
        for(j=0; j<NUM_VISIBLE+1; j++){
            neg_visible_activations[i][j] = 0;
            neg_visible_probs[i][j] = 0;
        }
    }


}

void train_rbm(int data_orig[TRAIN_SIZE][NUM_VISIBLE]){

    int e, t;
    int i,j;

    //Insert Bias into Data
    int data[TRAIN_SIZE][NUM_VISIBLE+1]; 
    for(t=0; t<TRAIN_SIZE; t++){
        for(j=0; j<NUM_VISIBLE+1; j++){
            if(j==0){
                data[t][j] = 1;
            }else{
                data[t][j] = data_orig[t][j-1];
            }
        }
    }

#ifdef DEBUG
    printf("\nTest Printing Data\n");
    for(i=0; i<TRAIN_SIZE; i++){
        for(j=0; j<NUM_VISIBLE+1; j++){
            printf("%d ", data[i][j]);
        }
        printf("\n");
    }   
#endif
 
    //Run Training for Desired number of epochs
    for(e=0; e<NUM_EPOCHS; e++){
        zero_arrays();
        //Calculate dot product of positive activations
        //Dot product of data and weights
        
        double ERROR = 0;

        for(t=0; t<TRAIN_SIZE; t++){
            //Calculate Pos Hidden Activations
            for(i=0; i<NUM_HIDDEN+1; i++){
                for(j=0; j<NUM_VISIBLE+1; j++){
                    pos_hidden_activations[t][i] += data[t][j]*weights[j][i];
                }
            }

            //Calculate Hidden Probability
            for(i=0; i<NUM_HIDDEN+1; i++){
                pos_hidden_probs[t][i] = logistic(pos_hidden_activations[t][i]);
            }

            //Calculate Positive Associations
            //Dot product of data and positive hidden probabilities
            for(i=0; i<NUM_HIDDEN+1; i++){
                for(j=0; j<NUM_VISIBLE+1; j++){
                    pos_assoc[j][i] += data[t][j]*pos_hidden_probs[t][i];
                }
            }

            //Calculate Negative Visible Activations
            for(i=0; i<NUM_HIDDEN+1; i++){
                if(pos_hidden_probs[t][i] > rand_twister()){
                    for(j=0; j<NUM_VISIBLE+1; j++){
                        //Gibbs Sampling
                        neg_visible_activations[t][j] += weights[j][i];
                    }
                }
            }

            //Calculate Neg Visible Probability
            for(j=0; j<NUM_VISIBLE+1; j++){
                neg_visible_probs[t][j] = logistic(neg_visible_activations[t][j]);
                if(j==0){
                    neg_visible_probs[t][j] = 1;
                }
            }

            //Calculate Neg Hidden Activations
            for(i=0; i<NUM_HIDDEN+1; i++){
                for(j=0; j<NUM_VISIBLE+1; j++){
                    neg_hidden_activations[t][i] += 
                        neg_visible_probs[t][j] * weights[j][i];
                }
            }

            //Calculate Neg Hidden Probabilities
            for(i=0; i<NUM_HIDDEN+1; i++){
                neg_hidden_probs[t][i] = logistic(neg_hidden_activations[t][i]);
            }


            //Calculate Neg Associations
            for(i=0; i<NUM_HIDDEN+1; i++){
                for(j=0; j<NUM_VISIBLE+1; j++){
                    neg_assoc[j][i] += neg_visible_probs[t][j]*neg_hidden_probs[t][i];
                }
            }

            //Calculate Error
            for(j=0; j<NUM_VISIBLE+1; j++){
                ERROR += pow(data[t][j] - neg_visible_probs[t][j], 2);
            }

        } //Done Calculating Training Data

        //Recalculate Weights
        for(i=0; i<NUM_HIDDEN+1; i++){
            for(j=0; j<NUM_VISIBLE+1; j++){
                weights[j][i] += LEARN_RATE * 
                    ((pos_assoc[j][i]-neg_assoc[j][i])/TRAIN_SIZE);
            }
        }
        //print_weights();
        printf("Epoch %d - Error: %f\n", e, ERROR);

#ifdef DEBUG
        printf("\nPositive Hidden Activations\n");
        for(t=0; t<TRAIN_SIZE; t++){
            for(i=0; i<NUM_HIDDEN+1; i++){
                printf("%f ", pos_hidden_activations[t][i]);
            }
            printf("\n");
        }

        printf("\nPositive Hidden Probabilities\n");
        for(t=0; t<TRAIN_SIZE; t++){
            for(j=0; j<NUM_HIDDEN+1; j++){
                printf("%f ", pos_hidden_probs[t][j]);
            }
            printf("\n");
        }
 
        printf("\nPositive Associations\n");
        for(j=0; j<NUM_VISIBLE+1; j++){
            for(i=0; i<NUM_HIDDEN+1; i++){
                printf("%f ", pos_assoc[j][i]);
            }
            printf("\n");
        }

        printf("\nNegative Activations\n");
        for(t=0; t<TRAIN_SIZE; t++){
            for(i=0; i<NUM_VISIBLE+1; i++){
                printf("%f ", neg_visible_activations[t][i]);
            }
            printf("\n");
        }

        printf("\nNegative Visible Probabilities\n");
        for(t=0; t<TRAIN_SIZE; t++){
            for(j=0; j<NUM_VISIBLE+1; j++){
                printf("%f ",neg_visible_probs[t][j]);
            }
            printf("\n");
        }
        
        printf("\nNegative Hidden Activations\n");
        for(t=0; t<TRAIN_SIZE; t++){
            for(i=0; i<NUM_HIDDEN+1; i++){
                printf("%f ", neg_hidden_activations[t][i]);
            }
            printf("\n");
        }


        printf("\nNegative Hidden Probabilities\n");
        for(t=0; t<TRAIN_SIZE; t++){
            for(i=0; i<NUM_HIDDEN+1; i++){
                printf("%f ",neg_hidden_probs[t][i]);
            }
            printf("\n");
        }
       
        printf("\nNeg Assoc\n");
        for(j=0; j<NUM_VISIBLE+1; j++){
            for(i=0; i<NUM_HIDDEN+1; i++){
                printf("%f ", neg_assoc[j][i]);
            }
            printf("\n");
        }

#endif 
    }

}


int main (){
    //Toy set training Data
    int data[TRAIN_SIZE][NUM_VISIBLE] = 
        {{1,1,1,0,0,0}, {1,0,1,0,0,0}, {1,1,1,0,0,0}, 
        {0,0,1,1,1,0}, {0,0,1,1,0,0}, {0,0,1,1,1,0}};

    //Initialize Weights
    initialize_weights();
    //Do Training
    train_rbm(data);
   
    print_weights();

    return 0;
}
