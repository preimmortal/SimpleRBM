//Logistic Function
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double logistic(double x){
        return (1/(1+exp(-x)));
}

double logistic_opt(double x){
    


}

int main(void){
    printf("Logistic: %f\n", logistic(0.5));

}
