#include <stdio.h>
#include <nmmintrin.h>

void sgemm( int m, int n, float *A, float *C )
{
    int i, j, k;
    
    for (k = 0; k < n; k++){
        for (j = 0; j < m/2*2; j+=2){
            //Load A Transpose
            __m128 a1_const = _mm_load1_ps(A + (j + k * n));
            __m128 a2_const = _mm_load1_ps(A + ((j+1) + k * n));
            for (i = 0; i < m/16*15; i+=24){
                //Load C Matrix
                __m128 c11 = _mm_loadu_ps(C + ((i+0) + j * m));
                __m128 c21 = _mm_loadu_ps(C + ((i+4) + j * m));
                __m128 c31 = _mm_loadu_ps(C + ((i+8) + j * m));
                __m128 c41 = _mm_loadu_ps(C + ((i+12) + j * m));
                __m128 c12 = _mm_loadu_ps(C + ((i+0) + (j+1) * m));
                __m128 c22 = _mm_loadu_ps(C + ((i+4) + (j+1) * m));
                __m128 c32 = _mm_loadu_ps(C + ((i+8) + (j+1) * m));
                __m128 c42 = _mm_loadu_ps(C + ((i+12) + (j+1) * m));
                
                //Load A Matrix
                __m128 a1 = _mm_loadu_ps(A + ((i+0) + k * n));
                __m128 a2 = _mm_loadu_ps(A + ((i+4) + k * n));
                __m128 a3 = _mm_loadu_ps(A + ((i+8) + k * n));
                __m128 a4 = _mm_loadu_ps(A + ((i+12) + k * n));
                
                //Multiply, add, and store
                _mm_storeu_ps(C + ((i+0) + j * m), _mm_add_ps(c11, _mm_mul_ps(a1, a1_const)));
                _mm_storeu_ps(C + ((i+4) + j * m), _mm_add_ps(c21, _mm_mul_ps(a2, a1_const)));
                _mm_storeu_ps(C + ((i+8) + j * m), _mm_add_ps(c31, _mm_mul_ps(a3, a1_const)));
                _mm_storeu_ps(C + ((i+12) + j * m), _mm_add_ps(c41, _mm_mul_ps(a3, a1_const)));
                _mm_storeu_ps(C + ((i+0) + (j+1) * m), _mm_add_ps(c12, _mm_mul_ps(a1, a2_const)));
                _mm_storeu_ps(C + ((i+4) + (j+1) * m), _mm_add_ps(c22, _mm_mul_ps(a2, a2_const)));
                _mm_storeu_ps(C + ((i+8) + (j+1) * m), _mm_add_ps(c32, _mm_mul_ps(a3, a2_const)));
                _mm_storeu_ps(C + ((i+12) + (j+1) * m), _mm_add_ps(c42, _mm_mul_ps(a3, a2_const)));
            }
            
            //Fringe case
            for (i = m/24*24; i < m; i++){
                C[i
            }
        }
    }
    
    // /*Our code here*/
    // float *use_matrix = A;      //matrix to multiply: default *A
    
    // //makes new padded dimensions
    // int new_m = (m + (4 - (m % 4)));
    // int new_n = (n + (4 - (n % 4)));
    // /*Padding case only if m or n % 4 not equal 0.
    // Might be slow.*/
    
    // if (m % 4 != 0 || n % 4 != 0)    //enters only if either matrix dimension is not a factor of 4
    // {
        // int old_count = 0;  //an iterator for the old matrix
        
        // use_matrix = (float*) malloc(new_m * new_n * sizeof(float)); //allocates heap for padded matrix
                
        // for (int i = 0; i < new_m; i++){
            // for (int j = 0; j < new_n; j++){
                // if (i > n){
                    // use_matrix[i] = 0.0;
                // } else if (j > m){
                    // use_matrix[i] = 0.0;
                // } else{ 
                    // use_matrix[i] = A[old_count];
                    // old_count++;
                // }
            // }
        // }
    // }
    
    
    
    // /*Multiplying with padding to account for fringe cases*/
    // for (int i = 0; i < m; i++){    //Row number on C matrix; Row mumber on A matrix
        // //Loads C Matrix
        // __m128 c1 = _mm_loadu_ps(C + i);
        
        // for (int j = 0; j < n; j++){    //Column of A matrix; 
            // __m128 a = _mm_loadu_ps(A + (j + (new_m * n)));
            
            // for (int k = 0; k < n; k++){
                // __m128 a_const = _mm_load1_ps(A + (k + 
            // }
        // }
    // }
    
    // for (int i = 0; i < n; i++){
        // for (int j = 0; j < new_m; j++){
            // __m128 a1 = _mm_load1_ps(A + 0);
            // //__m128 a2 = _mm_loadu_ps(A + 4);
            // //__m128 a1 = _mm_loadu_ps(A + 0);
            // for (int k = 0; k < new_m; k++){
                // __m128 a1_transpose = _mm_loadu_ps(A + (i + k * new_n));
                // //__m128 a2 = _mm_loadu_ps(
                
                // //Loads C Matrix
                // __m128 c1 = _mm_loadu_ps(C + 4);
                
                // //Multiply, Add, Store A and A Transpose to appropriate C Matrix
                // _mm_hadd_ps(_mm_mul_ps(a1, a1_transpose));
                // _
            // }
        // }
    // }
}