#include <stdio.h>
#include <nmmintrin.h>

void sgemm( int m, int n, float *A, float *C ){
    int i,j,k;
    
    for (j = 0; j < m; j++){
        for (i = 0; i < m/28*28; i+=28){
            //Load C Matrix
            __m128 c1 = _mm_loadu_ps(C + ((i+0) + j * m));
            __m128 c2 = _mm_loadu_ps(C + ((i+4) + j * m));
            __m128 c3 = _mm_loadu_ps(C + ((i+8) + j * m));
            __m128 c4 = _mm_loadu_ps(C + ((i+12) + j * m));
            __m128 c5 = _mm_loadu_ps(C + ((i+16) + j * m));
            __m128 c6 = _mm_loadu_ps(C + ((i+20) + j * m));
            __m128 c7 = _mm_loadu_ps(C + ((i+24) + j * m));
            
            for (k = 0; k < n; k++){
                //Load A Matrix
                __m128 a1 = _mm_loadu_ps(A + ((i+0) + k * m));
                __m128 a2 = _mm_loadu_ps(A + ((i+4) + k * m));
                __m128 a3 = _mm_loadu_ps(A + ((i+8) + k * m));
                __m128 a4 = _mm_loadu_ps(A + ((i+12) + k * m));
                __m128 a5 = _mm_loadu_ps(A + ((i+16) + k * m));
                __m128 a6 = _mm_loadu_ps(A + ((i+20) + k * m));
                __m128 a7 = _mm_loadu_ps(A + ((i+24) + k * m));
                
                //Load A Transpose Scalar
                __m128 a1_const = _mm_load1_ps(A + (j + k * m));
                
                //Multiply and Add
                c1 = _mm_add_ps(c1, _mm_mul_ps(a1, a1_const));
                c2 = _mm_add_ps(c2, _mm_mul_ps(a2, a1_const));
                c3 = _mm_add_ps(c3, _mm_mul_ps(a3, a1_const));
                c4 = _mm_add_ps(c4, _mm_mul_ps(a4, a1_const));
                c5 = _mm_add_ps(c5, _mm_mul_ps(a5, a1_const));
                c6 = _mm_add_ps(c6, _mm_mul_ps(a6, a1_const));
                c7 = _mm_add_ps(c7, _mm_mul_ps(a7, a1_const));
            }
            
            //Store
            _mm_storeu_ps(C + ((i+0) + j * m), c1);
            _mm_storeu_ps(C + ((i+4) + j * m), c2);
            _mm_storeu_ps(C + ((i+8) + j * m), c3);
            _mm_storeu_ps(C + ((i+12) + j * m), c4);
            _mm_storeu_ps(C + ((i+16) + j * m), c5);
            _mm_storeu_ps(C + ((i+20) + j * m), c6);
            _mm_storeu_ps(C + ((i+24) + j * m), c7);
        }
        
        int remainder = m - i;
        /*Remaining Fringe Cases*/
        
        if (remainder >= 24){
            //Load C Matrix
            __m128 c1 = _mm_loadu_ps(C + ((i+0) + j * m));
            __m128 c2 = _mm_loadu_ps(C + ((i+4) + j * m));
            __m128 c3 = _mm_loadu_ps(C + ((i+8) + j * m));
            __m128 c4 = _mm_loadu_ps(C + ((i+12) + j * m));
            __m128 c5 = _mm_loadu_ps(C + ((i+16) + j * m));
            __m128 c6 = _mm_loadu_ps(C + ((i+20) + j * m));
            
                for (k = 0; k < n; k++){
                    //Load A Matrix
                    __m128 a1 = _mm_loadu_ps(A + ((i+0) + k * m));
                    __m128 a2 = _mm_loadu_ps(A + ((i+4) + k * m));
                    __m128 a3 = _mm_loadu_ps(A + ((i+8) + k * m));
                    __m128 a4 = _mm_loadu_ps(A + ((i+12) + k * m));
                    __m128 a5 = _mm_loadu_ps(A + ((i+16) + k * m));
                    __m128 a6 = _mm_loadu_ps(A + ((i+20) + k * m));
                    
                    //Load A Transpose Scalar
                    __m128 a1_const = _mm_load1_ps(A + (j + k * m));
                    
                    //Multiply and Add
                    c1 = _mm_add_ps(c1, _mm_mul_ps(a1, a1_const));
                    c2 = _mm_add_ps(c2, _mm_mul_ps(a2, a1_const));
                    c3 = _mm_add_ps(c3, _mm_mul_ps(a3, a1_const));
                    c4 = _mm_add_ps(c4, _mm_mul_ps(a4, a1_const));
                    c5 = _mm_add_ps(c5, _mm_mul_ps(a5, a1_const));
                    c6 = _mm_add_ps(c6, _mm_mul_ps(a6, a1_const));
                }
            
            //Store
            _mm_storeu_ps(C + ((i+0) + j * m), c1);
            _mm_storeu_ps(C + ((i+4) + j * m), c2);
            _mm_storeu_ps(C + ((i+8) + j * m), c3);
            _mm_storeu_ps(C + ((i+12) + j * m), c4);
            _mm_storeu_ps(C + ((i+16) + j * m), c5);
            _mm_storeu_ps(C + ((i+20) + j * m), c6);
            
            i+=24;
        } else if (remainder >= 20){
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
            
            i+=20;
        } else if (remainder >= 16){
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
            
            i+=16;
        } else if (remainder >= 12){
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
            
            i+=12;
        } else if (remainder >= 8){
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
            
            i+=8;
        } else if (remainder >= 4){
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
            
            i+=4;
        }
        
        //Remaining non multiples of 4
        for (i = i; i < m; i++){
            for (k = 0; k < n; k++){
                C[i + j * m] += A[i + k * m] * A[j + k * m];
            }
        }
    }
}