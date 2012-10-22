#include <stdio.h>
#include <nmmintrin.h>

void sgemm( int m, int n, float *A, float *C ){
    int i,j,k
    
    for (j = 0; j < m; k++){
        for (i = 0; i < m/20*20; i+=20){
            //Load C Matrix
            __m128 c1 = _mm_loadu_ps(C + ((i+0) + j * m));
            __m128 c2 = _mm_loadu_ps(C + ((i+4) + j * m));
            __m128 c3 = _mm_loadu_ps(C + ((i+8) + j * m));
            __m128 c4 = _mm_loadu_ps(C + ((i+12) + j * m));
            __m128 c5 = _mm_loadu_ps(C + ((i+16) + j * m));
            
            for (k = 0; k < n; k++){
                //Load A Matrix
                __m128 a1 = _mm_loadu_ps(A + ((i+0) + k * m));
                __m128 a2 = _mm_loadu_ps(A + ((i+4) + k * m));
                __m128 a3 = _mm_loadu_ps(A + ((i+8) + k * m));
                __m128 a4 = _mm_loadu_ps(A + ((i+12) + k * m));
                __m128 a5 = _mm_loadu_ps(A + ((i+16) + k * m));
                
                //Load A Transpose Scalar
                __m128 a1_const = _mm_load1_ps(A + (j + k * m));
                
                //Multiply and Add
                c1 = _mm_add_ps(c1, _mm_mul_ps(a1, a1_const));
                c2 = _mm_add_ps(c2, _mm_mul_ps(a2, a1_const));
                c3 = _mm_add_ps(c3, _mm_mul_ps(a3, a1_const));
                c4 = _mm_add_ps(c4, _mm_mul_ps(a4, a1_const));
                c5 = _mm_add_ps(c5, _mm_mul_ps(a5, a1_const));
            }
            
            //Store
            _mm_storeu_ps(C + ((i+0) + j * m), c1);
            _mm_storeu_ps(C + ((i+4) + j * m), c2);
            _mm_storeu_ps(C + ((i+8) + j * m), c3);
            _mm_storeu_ps(C + ((i+12) + j * m), c4);
            _mm_storeu_ps(C + ((i+16) + j * m), c5);
        }
        
        /*Remaining Fringe Cases*/
        for (i = m/20*20; i < m/16*16; i+=16){
            //Load C Matrix
            __m128 c1 = _mm_loadu_ps(C + ((i+0) + j * m));
            __m128 c2 = _mm_loadu_ps(C + ((i+4) + j * m));
            __m128 c3 = _mm_loadu_ps(C + ((i+8) + j * m));
            __m128 c4 = _mm_loadu_ps(C + ((i+12) + j * m));
            
            for (k = 0; k < n; k++){
                //Load A Matrix
                __m128 a1 = _mm_loadu_ps(A + ((i+0) + k * m));
                __m128 a2 = _mm_loadu_ps(A + ((i+4) + k * m));
                __m128 a3 = _mm_loadu_ps(A + ((i+8) + k * m));
                __m128 a4 = _mm_loadu_ps(A + ((i+12) + k * m));
                
                //Load A Transpose Scalar
                __m128 a1_const = _mm_load1_ps(A + (j + k * m));
                
                //Multiply and Add
                c1 = _mm_add_ps(c1, _mm_mul_ps(a1, a1_const));
                c2 = _mm_add_ps(c2, _mm_mul_ps(a2, a1_const));
                c3 = _mm_add_ps(c3, _mm_mul_ps(a3, a1_const));
                c4 = _mm_add_ps(c4, _mm_mul_ps(a4, a1_const));
            }
            
            //Store
            _mm_storeu_ps(C + ((i+0) + j * m), c1);
            _mm_storeu_ps(C + ((i+4) + j * m), c2);
            _mm_storeu_ps(C + ((i+8) + j * m), c3);
            _mm_storeu_ps(C + ((i+12) + j * m), c4);
        }
        
        for (i = m/16*16; i < m/12*12; i+=12){
            //Load C Matrix
            __m128 c1 = _mm_loadu_ps(C + ((i+0) + j * m));
            __m128 c2 = _mm_loadu_ps(C + ((i+4) + j * m));
            __m128 c3 = _mm_loadu_ps(C + ((i+8) + j * m));
            
            for (k = 0; k < n; k++){
                //Load A Matrix
                __m128 a1 = _mm_loadu_ps(A + ((i+0) + k * m));
                __m128 a2 = _mm_loadu_ps(A + ((i+4) + k * m));
                __m128 a3 = _mm_loadu_ps(A + ((i+8) + k * m));
                
                //Load A Transpose Scalar
                __m128 a1_const = _mm_load1_ps(A + (j + k * m));
                
                //Multiply and Add
                c1 = _mm_add_ps(c1, _mm_mul_ps(a1, a1_const));
                c2 = _mm_add_ps(c2, _mm_mul_ps(a2, a1_const));
                c3 = _mm_add_ps(c3, _mm_mul_ps(a3, a1_const));
            }
            
            //Store
            _mm_storeu_ps(C + ((i+0) + j * m), c1);
            _mm_storeu_ps(C + ((i+4) + j * m), c2);
            _mm_storeu_ps(C + ((i+8) + j * m), c3);
        }
        
        for (i = m/12*12; i < m/8*8; i+=8){
            //Load C Matrix
            __m128 c1 = _mm_loadu_ps(C + ((i+0) + j * m));
            __m128 c2 = _mm_loadu_ps(C + ((i+4) + j * m));
            
            for (k = 0; k < n; k++){
                //Load A Matrix
                __m128 a1 = _mm_loadu_ps(A + ((i+0) + k * m));
                __m128 a2 = _mm_loadu_ps(A + ((i+4) + k * m));
                
                //Load A Transpose Scalar
                __m128 a1_const = _mm_load1_ps(A + (j + k * m));
                
                //Multiply and Add
                c1 = _mm_add_ps(c1, _mm_mul_ps(a1, a1_const));
                c2 = _mm_add_ps(c2, _mm_mul_ps(a2, a1_const));
            }
            
            //Store
            _mm_storeu_ps(C + ((i+0) + j * m), c1);
            _mm_storeu_ps(C + ((i+4) + j * m), c2);
        }
        
        for (i = m/8*8; i < m/4*4; i+=4){
            //Load C Matrix
            __m128 c1 = _mm_loadu_ps(C + ((i+0) + j * m));
            
            for (k = 0; k < n; k++){
                //Load A Matrix
                __m128 a1 = _mm_loadu_ps(A + ((i+0) + k * m));
                
                //Load A Transpose Scalar
                __m128 a1_const = _mm_load1_ps(A + (j + k * m));
                
                //Multiply and Add
                c1 = _mm_add_ps(c1, _mm_mul_ps(a1, a1_const));
            }
            
            //Store
            _mm_storeu_ps(C + ((i+0) + j * m), c1);
        }
        
        for (i = m/4*4; i < m; i++){
            C[i + j * n] += A[i + k * m] * A[j + k * m];
        }
    }
}