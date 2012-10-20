#include <stdio.h>
#include <nmmintrin.h>

void sgemm( int m, int n, float *A, float *C )
{
    /*Our code here*/
    float *use_matrix = A;      //matrix to multiply: default *A
    
    //makes new padded dimensions
    int new_m = (m + (4 - (m % 4)));
    int new_n = (n + (4 - (n % 4)));
    /*Padding case only if m or n % 4 not equal 0.
    Might be slow.*/
    
    if(m % 4 != 0 || n % 4 != 0)    //enters only if either matrix dimension is not a factor of 4
    {
        int old_count = 0;  //an iterator for the old matrix
        
        use_matrix = (float*) malloc(new_m * new_n * sizeof(float)); //allocates heap for padded matrix
                
        for (int i = 0; i < new_m; i++){
            for (int j = 0; j < new_n; j++){
                if (i > n){
                    use_matrix[i] = 0.0;
                } else if (j > m){
                    use_matrix[i] = 0.0;
                } else{ 
                    use_matrix[i] = A[old_count];
                    old_count++;
                }
            }
        }
    }
    
    /*Multiplying with padding to account for fringe cases*/
    for (int i = 0; i < n; i++){ //outer loop
        for (int j = 0; j < new_m; j++){
            __m128 a1 = _mm_loadu_ps(A + 0);
            //__m128 a2 = _mm_loadu_ps(A + 4);
            //__m128 a1 = _mm_loadu_ps(A + 0);
            for (int k = 0; k < new_m; k++){
                __m128 a_transpose1 = _mm_loadu_ps(A + (i + k * new_n));
                //__m128 a2 = _mm_loadu_ps(
                
                //Loading C Matrix
                __128 c1 = _mm_loadu_ps(C + 4);
                
                
            }
        }
    }
}