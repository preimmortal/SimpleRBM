#include <stdio.h>
#include <stdlib.h>

#define SEED 12345
static unsigned int s1 = SEED, s2 = SEED, s3 = SEED, b;

float taus_rng ()
{   /* Generates numbers between 0 and 1. */
    b = (((s1 << 13) ^ s1) >> 19);
    s1 = (((s1 & 4294967294) << 12) ^ b);
    b = (((s2 << 2) ^ s2) >> 25);
    s2 = (((s2 & 4294967288) << 4) ^ b);
    b = (((s3 << 3) ^ s3) >> 11);
    s3 = (((s3 & 4294967280) << 17) ^ b);
    return ((s1 ^ s2 ^ s3) * 2.3283064365386963e-10);
}


int main(void){
    int i;
    for(i=0; i<100; i++){
        printf("Random: %f\n",taus_rng());
    }
    return 0;
}
