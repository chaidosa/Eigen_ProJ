/**
 * The following Program is for computing eigen values using QR algorithm
 * by using HouseHolder transformations for decomposition.
 * 
 * Reference: https://en.wikipedia.org/wiki/QR_decomposition , https://en.wikipedia.org/wiki/QR_algorithm .
 * 
 *  Library used for computation cblas http://www.netlib.org/blas/
 * */

#include <iostream>
#include <bits/stdc++.h>
#include<sys/time.h> 

extern "C"
{
     #include <cblas.h>
}

using namespace std;
//global const variable

const int size=126;

//Function definitions
void res_householder_tr(double H[], double A[], int col);
void give_me_identity(double I[], int N);
void print_mat(double mat[], int n);
void initialize_zero(double P[], int n);
void QrDec(double A[], double Q[], double R[]);


//decompooses the matrix a into Q and R.
void QrDec(double A[], double Q[], double R[]){ 
    initialize_zero(Q, size);
    initialize_zero(R, size);

    //Making Q to be identity matrix
    give_me_identity(Q, size);

    //calling householder transformation for the required column
    for(int col=0;col<size-1;col++){
        double B[size*size];
        give_me_identity(B, size);
        int N=size-col;
        double H[N*N];

        res_householder_tr(H, A, col);      
        
        //copying the transformed elements
        for(int row1=col,row2=0;row1<size;row1++,row2++){
            for(int col1=col,col2=0;col1<size;col1++,col2++){
                B[row1*size+col1]=H[row2*N+col2];
            }
        }
       
        //Q=Q*B
        double temp[size*size];
        initialize_zero(temp,size);
        cblas_dgemm(CblasRowMajor,  CblasNoTrans,  CblasNoTrans, size, size,size, 1.0, Q, size, B, size, 0.0, temp, size);
        for(int row=0;row<size;row++)
        {
            for(int col=0;col<size;col++){
                Q[row*size+col]=temp[row*size+col];
            }
        }
        //A=B*A
        initialize_zero(temp, size);
        cblas_dgemm(CblasRowMajor,  CblasNoTrans,  CblasNoTrans, size, size,size, 1.0, B, size, A, size, 0.0, temp, size);
        for(int row=0;row<size;row++)
        {
            for(int col=0;col<size;col++){
                A[row*size+col]=temp[row*size+col];
            }
        }


    }          
    //R=A
    for(int row=0;row<size;row++){
        for(int col=0;col<size;col++){
            R[row*size+col]=A[row*size+col];
        }
    }
    

}


//Householder Transformation
void res_householder_tr(double H[], double A[], int col){
    int N= size-col;
    double v1[N],eye[N];

    for(int i=0;i<N;i++){
        v1[i]=0;
        eye[i]=0;
    }
   
    //copying the desired column from the matrix
    for(int i=col,j=0;i<size,j<N;i++,j++){      
        v1[j]=A[i*size+col];      
    }
    
    
    double v1_norm=0.0;
    v1_norm= cblas_dnrm2(N,v1,1); 
    eye[0]=v1_norm;

    for(int row=0;row<N;row++)
    {
        v1[row]=eye[row]-v1[row];
    }


    v1_norm= cblas_dnrm2(N,v1,1); 
   
    /**
     * V = U/||U||
    */
    for(int row=0;row<N;row++){
        v1[row]=v1[row]/v1_norm;
    }

    //identity Matrix
    double Idn[N*N];
    give_me_identity(Idn,N);
   
    //vector vector multiplication result will be here.
    double vvm[N*N];

    //vvm= V*V'
    cblas_dgemm(CblasRowMajor,  CblasNoTrans,  CblasNoTrans, N, N,1, 1.0, v1, 1, v1, N, 0.0, vvm, N);
   
    /**
     * According to householder formula
     * P=I-2*V*V'
     * here P is H
     */
    for(int row=0;row<N;row++){
        for(int col=0;col<N;col++){
            H[row*N+col]=Idn[row*N+col]-2*vvm[row*N+col];
        }
    }
    
}


//returns an identity matrix 
void give_me_identity(double I[],int N){
    for(int row=0;row<N;row++){
        for(int col=0;col<N;col++){
            I[row*N+col]=0;
          
        }
    }
    for(int i=0;i<N;i++){
        I[i*N+i]=1;
    }
}


//Prints the given matrix
void print_mat(double mat[],int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<mat[i*n+j]<<" ";
        }
        cout<<"\n";
    }
}


//return matrix with all entries to be zero
void initialize_zero(double P[],int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            P[i*n+j]=0;
        }
    }
}


int main(){
    //For input and output.
    #ifndef INPUT_OUT
	freopen("input.txt","r",stdin);
	freopen("output.txt","w",stdout);
    #endif
    //initializing all elements to zero 
    double Q[size*size];
    double R[size*size];
    double A[size*size];

    initialize_zero(Q,size);
    initialize_zero(R,size);    
    initialize_zero(A,size);

    //input of matrix
    for(int rows=0;rows<size;rows++){
        for(int col=0;col<size;col++){
            cin>>A[rows*size+col];
        }
    } 
    //For measuring execution time
    timeval start,end;
    double time_taken;


    //Iterating for 20 times this step can be done differently
    gettimeofday(&start,NULL);
    for(int i=0;i<20;i++)
    {
        QrDec(A,Q,R);

        //A=R*Q;     
        cblas_dgemm(CblasRowMajor,  CblasNoTrans,  CblasNoTrans, size, size,size, 1.0, R, size, Q, size, 0.0, A, size);
    }
   

   cout<<"Eigen Values are:\n";
    for(int diag=0;diag<size;diag++){
        cout<<A[diag*size+diag]<<endl;
    }
    gettimeofday(&end,NULL);
    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec -start.tv_usec)) * 1e-6;
	cout<<"Execution time "<<time_taken<<endl;
   return 0; 
}
