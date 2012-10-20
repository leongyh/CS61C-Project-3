#include <stdio.h>
#include <emmintrin.h>

void sgemm( int m, int n, float *A, float *C )
{
    /*Our code here*/
    
    //Padding case only if m or n % 4 not equal 0:
    float *padded_matrix;
    
    if(m % 4 != 0 || n % 4 != 0)
    {
        int new_m = (m + (4 - (m % 4)));
        int new_n = (n + (4 - (n % 4)));
        int old_count = 0;
        
        padded_matrix = (float*) malloc(new_m * new_n * sizeof(float));
                
        for (int i = 0; i < new_m; i++){
            for (int j = 0; j < new_n; j++){
                if (i > n){
                    padded_matrix[i] = 0.0;
                } else if (j > m){
                    padded_matrix[i] = 0.0;
                } else{ 
                    padded_matrix[i] = A[old_count];
                    old_count++;
                }
            }
        }
    }
    
    for (int i = 0; i < n; i = i + 4){
        __m128 c_matrix = _mm_loadu_ps(C + i);
        
    }
}