/**
 * The following code is for calculating eigen vector using Power iteration method.
 * The program accepts input matrices of size 126x126 only for now.
 * Reference: Numerical method in Engineering with Matlab by Jaan Kiusalaas
 * declared matrices outside main to avoid segmentation fault.
 */ 
#include<cmath>
#include <iomanip> 
#include<bits/stdc++.h>
#include<omp.h>
using namespace std;

const int size = 4736;
double trMatrix[size*size] = {0};
double vec[size]={0};
double vOld[size]={0},z[size]={0};
int main(){
#ifndef Input_Output
	freopen("input.txt","r",stdin);
	freopen("output.txt","w",stdout);
#endif

 	int iteration=1;	
	double start,end;
	start = omp_get_wtime();
	double zMag=0;
	double eigenval,eigNew,eigOld=1,tolError;
	int vSign=1;
		
	//Adding transformation matrix
	for(unsigned int i=0;i<size;i++)
	{
		for(unsigned j=0;j<size;j++)
		{
			cin>>trMatrix[i*size+j];
		}
	}
	//Initial guess vector to be unit vector.
	#pragma omp parallel
	vec[0]=1;
	eigOld = 1.0; //hardcoded to 1 because of the initial guess: magnitude(unit vector).
	
	//Multiplication of Transformation matrix and initial guess vector
	#pragma omp parallel 
	{
		unsigned int i,j;
		double temp=0.0;
	#pragma omp for	
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{
			temp += trMatrix[i*size+j]*vec[j];
		}
	#pragma omp critical
	{
		z[i]=temp;
		temp=0;
	}	
	}
	}
	//magnitude of new vector
	zMag = 0.0;
	#pragma omp parallel for reduction(+ : zMag)	
	for(unsigned int i=0;i<size;i++)
	{
		zMag += z[i]*z[i];
	}	
	zMag = sqrt(zMag);
	eigNew = zMag;

	//Normalization
	#pragma omp parallel for
	for(unsigned int i=0;i<size;i++)
	{
		vec[i] = z[i]/zMag;
	}

	tolError = fabs(eigOld-eigNew);
		
	//Iterating till change is negliglible
	while (tolError > 1.0e-12)
	{		
		//Initailizaing vOld 
		//#pragma omp parallel for
		for(unsigned int i=0;i<size;i++)
		{
			vOld[i]=vec[i];
		}

	//Multiplication of Transformation matrix and initial guess vector	

	#pragma omp parallel 
	{
		unsigned int i,j;
		double temp=0.0;
		#pragma omp for 
		for(i=0;i<size;i++)
		{	
			for(j=0;j<size;j++)
			{   
				temp += trMatrix[i*size+j]*vec[j];
			}
		#pragma omp critical
		{
			z[i]=temp;
			temp=0;
		}	
		}
	}

		zMag=0;
		//magnitude of new vector
		#pragma omp parallel for reduction(+ : zMag)
		for(unsigned int i=0;i<size;i++)
		{
			zMag += z[i]*z[i];
		}
		zMag = sqrt(zMag);

		//Normalization
		#pragma omp parallel for
		for(unsigned int i=0;i<size;i++)
		{
			vec[i] = z[i]/zMag;
		}
		
		//Sign of the vector
		int temp1=0;
		#pragma omp parallel for reduction(+ : temp1)
		for(unsigned int i=0;i<size;i++)
		{
			temp1 = temp1+vOld[i]*vec[i];
		}
		if(temp1<0)
			vSign = -1;
		else
			vSign = 1;
		
		//Multiplying vector with sign
		#pragma omp parallel for
		for(unsigned int i=0;i<size;i++)
		{
			vec[i]=vec[i]*vSign;
		}

		//Checking the tolerable error
		eigNew = zMag;
		tolError = fabs(eigOld-eigNew);
		eigOld=eigNew;
		++iteration;

		if(iteration==2000) //if transformation is applied too many times.
			break;
	}
end = omp_get_wtime();
	eigenval = vSign*eigNew;
	cout<<std::setprecision(16)<<eigenval<<endl;
	cout<<"Iteration: "<<iteration<<"\n";
	cout<<"time "<<end-start;
	return 0;
}
