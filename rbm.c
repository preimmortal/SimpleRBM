//Simple RBM Implementation

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "rbm.h"

#define NUM_HIDDEN 2
#define NUM_VISIBLE 6
#define TRAIN_SIZE 6
#define NUM_EPOCHS 1
#define LEARN_RATE 0.1

#define DEBUG

//Weights Vector with Bias Units
//in first col/row
double weights[NUM_VISIBLE+1][NUM_HIDDEN+1];

double pos_hidden_activations[TRAIN_SIZE][NUM_HIDDEN+1];
double pos_hidden_probs[TRAIN_SIZE][NUM_HIDDEN+1];
double pos_assoc[NUM_VISIBLE+1][NUM_HIDDEN+1];



double logistic(double x){
    return (1/(1+exp(-x)));
}

void initialize_weights(){
    int i,j;
    for(i=0;i<NUM_VISIBLE+1;i++){
        for(j=0; j<NUM_HIDDEN+1;j++){
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

void zero_arrays(){
    int i, j;
    //Zero Out Pos Arrays
    for(i=0;i<TRAIN_SIZE;i++){
        for(j=0;j<NUM_HIDDEN+1;j++){
            pos_hidden_activations[i][j] = 0;
            pos_hidden_probs[i][j] = 0;
        }
    }
    for(i=0; i<NUM_VISIBLE+1; i++){
        for(j=0; j<NUM_HIDDEN+1; j++){
            pos_assoc[i][j] = 0;
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
        for(t=0; t<TRAIN_SIZE; t++){
            for(i=0; i<NUM_HIDDEN+1; i++){
                for(j=0; j<NUM_VISIBLE+1; j++){
                    pos_hidden_activations[t][i] += data[t][j]*weights[j][i];
                }
            }
        }

        //Calculate Hidden Probability
        for(t=0; t<TRAIN_SIZE; t++){
            for(j=0; j<NUM_HIDDEN+1; j++){
                pos_hidden_probs[t][j] = logistic(pos_hidden_activations[t][j]);
            }
        }
    
        //Calculate Positive Associations
        //Dot product of data and positive hidden probabilities
        for(j=0; j<NUM_VISIBLE+1; j++){
            for(i=0; i<NUM_HIDDEN+1; i++){
                for(t=0; t<TRAIN_SIZE; t++){
                    pos_assoc[j][i] += data[t][j]*pos_hidden_probs[t][i];
                }
            }
        }

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
    

    return 0;
}
