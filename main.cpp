/*
 * Copyright (c) University of Zagreb, Faculty of Electrical Engineering and Computing
 * Authors: Nikola Banic <nikola.banic@fer.hr> and Sven Loncaric <sven.loncaric@fer.hr>
 * 
 * This is only a research code and is therefore only of prototype quality.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * LITERATURE:
 * 
 * N. Banic and S. Loncaric
 * "Light Random Sprays Retinex: Exploiting the Noisy Illumination Estimation"
 * 
 */

#include <cstdio>
#include <cmath>

#if defined(_WIN32) || defined(_WIN64)
#include <cv.h>
#include <highgui.h>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

#define CV_LOAD_IMAGE_ANYDEPTH 2
#define CV_LOAD_IMAGE_ANYCOLOR 4

/**
	Filters an image with double precision data using an averagig kernel of given size.
	
	@param[in]	img Image to be filtered.
	@param[out] result The filtered image.
	@param[in]	k Averaging kernel size.
 */
void Filter64F(cv::Mat img, cv::Mat &result, int k){
	
	int rows=img.rows;
	int cols=img.cols;
	
	int cn=img.channels();
	
	cv::Mat currentResult=cv::Mat::zeros(rows, cols, CV_64FC3);
	
	cv::Vec3d *data=(cv::Vec3d *)img.data;
	
	double *s=new double[(rows+1)*(cols+1)*cn];
	
	s[1*(cols+1)*cn+1*cn+0]=(*data)[0];
	s[1*(cols+1)*cn+1*cn+1]=(*data)[1];
	s[1*(cols+1)*cn+1*cn+2]=(*data)[2];
	
	for (int i=0;i<rows+1;++i){
		for (int j=0;j<cn;++j){
			s[i*(cols+1)*cn+0*cn+j]=0;
		}
	}
	
	for (int i=0;i<cols+1;++i){
		for (int j=0;j<cn;++j){
			s[0*(cols+1)*cn+i*cn+j]=0;
		}
	}
	
	for (int i=0;i<rows;++i){
		for (int j=0;j<cols;++j){
			cv::Vec3d pixel=*(data+i*cols+j);
			for (int k=0;k<3;++k){
				s[(i+1)*(cols+1)*cn+(j+1)*cn+k]=pixel[k]-s[i*(cols+1)*cn+j*cn+k]+s[i*(cols+1)*cn+(j+1)*cn+k]+s[(i+1)*(cols+1)*cn+j*cn+k];
			}
		}
	}
	
	cv::Vec3d *output=(cv::Vec3d *)currentResult.data;
	for (int ch=0;ch<cn;++ch){
		for (int i=0;i<rows;++i){
			int row=i+1;
			
			int startRow=row-(k-1)/2-1;
			if (startRow<0){
				startRow=0;
			}
			
			int endRow=row+k/2;
			if (endRow>rows){
				endRow=rows;
			}
			
			for (int j=0;j<cols;++j){
				int col=j+1;
				int startCol=col-(k-1)/2-1;
				if (startCol<0){
					startCol=0;
				}
				
				int endCol=col+k/2;
				if (endCol>cols){
					endCol=cols;
				}
				cv::Vec3d &r=*(output+i*cols+j);
				r[ch]=(s[endRow*(cols+1)*cn+endCol*cn+ch]-s[endRow*(cols+1)*cn+startCol*cn+ch]-s[startRow*(cols+1)*cn+endCol*cn+ch]+s[startRow*(cols+1)*cn+startCol*cn+ch])/((endRow-startRow)*(endCol-startCol));
				
			}
			
		}
	}
	currentResult.copyTo(result);
	
	delete[] s;
	
}

/**
	Creates random sprays that are used for determining the neighbourhood.
	
	@param[in]	spraysCount Number of sprays to create.
	@param[in]	spraySize Size of individual spray in pixels.
	@return Returns the pointer to the created sprays.
 */
cv::Point2i **CreateSprays(int spraysCount, int n, int R){
	
	cv::RNG random;

	cv::Point2i **sprays=new cv::Point2i*[spraysCount];
	for (int i=0;i<spraysCount;++i){
		sprays[i]=new cv::Point2i[n];
		for (int j=0;j<n;++j){
			
			double angle=2*CV_PI*random.uniform(0.0, 1.0);
			double r=R*random.uniform(0.0, 1.0);

			sprays[i][j].x=r*cos(angle);
			sprays[i][j].y=r*sin(angle);
		}
	}
	
	return sprays;
}

/**
	Deletes previously created sprays.
	
	@param[in]	sprays Pointer to the sprays.
	@param[in]	spraysCount Number of sprays.
 */
void DeleteSprays(cv::Point2i **sprays, int spraysCount){
	
	for (int i=0;i<spraysCount;++i){
		delete[] sprays[i];
	}

	delete[] sprays;

}

/**
	Performs the Random Sprays Retinex algorithm on a given image for specified parameters.
	
	@param[in]	source The image to be processed.
	@param[out]	destination The resulting image.
	@param[in]	N Number of sprays to create.
	@param[in]	n Size of individual spray in pixels.
	@param[in]	upperBound Maximal value for a pixel channel.
	@param[in]	rowsStep Rows counting step.
	@param[in]	colsStep Columns counting step.
 */
void PerformRandomSpraysRetinex(cv::Mat source, cv::Mat &destination, int N, int n, double upperBound, int rowsStep, int colsStep){
	
	int rows=source.rows;
	int cols=source.cols;

	int R=sqrt((double)(rows*rows+cols*cols))+0.5;

	int spraysCount=1000*N;
	cv::Point2i **sprays=CreateSprays(spraysCount, n, R);
	
	cv::Mat normalized;
	source.convertTo(normalized, CV_64FC3);

	int outputRows=rows/rowsStep;
	int outputCols=cols/colsStep;
	destination=cv::Mat(outputRows, outputCols, CV_64FC3);

	cv::Vec3d *input=(cv::Vec3d *)normalized.data;
	cv::Vec3d *inputPoint=input;
	cv::Vec3d *output=(cv::Vec3d *)destination.data;
	cv::Vec3d *outputPoint=output;

	cv::RNG random;

	// cv::Mat certainity=cv::Mat::zeros(rows, cols, CV_64FC1);

	for (int outputRow=0;outputRow<outputRows;++outputRow){
		for (int outputCol=0;outputCol<outputCols;++outputCol){
			
			// Per default, rowsStep and colsStep are one.
			int row=outputRow*rowsStep;
			int col=outputCol*colsStep;

			inputPoint=input+row*cols+col;
			outputPoint=output+outputRow*outputCols+outputCol;
			
			cv::Vec3d &currentPoint=*inputPoint;
			cv::Vec3d &finalPoint=*outputPoint;
			finalPoint=cv::Vec3d(0, 0, 0);

			for (int i=0;i<N;++i){
				
				// Choose a random spray:
				int selectedSpray=random.uniform(0, spraysCount);
				cv::Vec3d max=cv::Vec3d(0, 0, 0);

                // Find the brightest point of all sprays, that is the
                // maximum values in all channels.
				for (int j=0;j<n;++j){
					
					int newRow=row+sprays[selectedSpray][j].y;
					int newCol=col+sprays[selectedSpray][j].x;

					if (newRow>=0 && newRow<rows && newCol>=0 && newCol<cols){
						
						cv::Vec3d &newPoint=input[newRow*cols+newCol];

						for (int k=0;k<3;++k){
							if (max[k]<newPoint[k]){
								max[k]=newPoint[k];
							}
						}
					}
				}

                // Normalize point, that is relate the point brightness to the brightest
                // point found on the sprays.
				for (int k=0;k<3;++k){
					finalPoint[k]+=currentPoint[k]/max[k];
				}
			}
			
			finalPoint/=N;

			for (int i=0;i<3;++i){
				if (finalPoint[i]>1){
					finalPoint[i]=1;
				}
			}

		}
	}

	double scaleFactor=upperBound;
	
	if (rowsStep>1 || colsStep>1){
		resize(destination, destination, source.size());
	}

	destination=destination*scaleFactor-1;

	destination.convertTo(destination, source.type());

	DeleteSprays(sprays, spraysCount);
	
}

/**
	Performs image enhancement using the Light Random Sprays Algorithm on a given image for specified parameters.
	
	@param[in]	source The image to be enhanced.
	@param[out]	destination The resulting image.
	@param[in]	N Number of sprays to create.
	@param[in]	n Size of individual spray in pixels.
	@param[in]	inputKernelSize The size of the kernel for blurring the original image and the RSR resulting image.
	@param[in]	inputSigma The input kernel sigma when using Gaussian kernels. If set to 0, the averaging kernel is used.
	@param[in]	intensityChangeKernelSize The size of the kernel for blurring the intensity change.
	@param[in]	intensityChangeSigma The intensity change kernel sigma when using Gaussian kernels. If set to 0, the averaging kernel is used.
	@param[in]	rowsStep Rows counting step.
	@param[in]	colsStep Columns counting step.
	@param[in]	normalizeIntensityChange The flag indicating wheather to normalize the intensity change (i. e. to perform only chromatic adaptation) or not (i. e. to perform chromatic adaptation and brightness adjustment).
	@param[in]	upperBound Maximal value for a pixel channel.
 */
void PerformLightRandomSpraysRetinex(cv::Mat source, cv::Mat &destination, int N, int n, int inputKernelSize, double inputSigma, int intensityChangeKernelSize, double intensityChangeSigma, int rowsStep, int colsStep, bool normalizeIntensityChange, double upperBound){
	
	cv::Mat inputSource;
	cv::Mat inputRetinex;
	cv::Mat retinex;

	PerformRandomSpraysRetinex(source, retinex, N, n, upperBound, rowsStep, colsStep);

	source.convertTo(inputSource, CV_64FC3);
	retinex.convertTo(inputRetinex, CV_64FC3);

	if (normalizeIntensityChange){
		cv::Mat illuminant;
		cv::divide(inputSource, inputRetinex, illuminant);
		std::vector<cv::Mat> illuminantChannels;
	
		split(illuminant, illuminantChannels);
		cv::Mat illuminantAverage=(illuminantChannels[0]+illuminantChannels[1]+illuminantChannels[2])/3;
		for (int i=0;i<3;++i){
			cv::divide(illuminantChannels[i], illuminantAverage, illuminantChannels[i]);
		}
		cv::merge(illuminantChannels, illuminant);
		
		inputSource=inputRetinex.mul(illuminant);
	}

	if (inputKernelSize>1){
		if (inputSigma==0.0){
			cv::Mat averaging=cv::Mat::ones(inputKernelSize, inputKernelSize, CV_64FC1)/(double)(inputKernelSize*inputKernelSize);
			Filter64F(inputSource, inputSource, inputKernelSize);
			Filter64F(inputRetinex, inputRetinex, inputKernelSize);
		} else{
			GaussianBlur(inputSource, inputSource, cv::Size(inputKernelSize, inputKernelSize), inputSigma);
			GaussianBlur(inputRetinex, inputRetinex, cv::Size(inputKernelSize, inputKernelSize), inputSigma);
		}
	}
	
	cv::Mat illuminant;
	divide(inputSource, inputRetinex, illuminant);
	std::vector<cv::Mat> illuminantChannels;
	
	if (intensityChangeKernelSize>1){
		if (intensityChangeSigma==0.0){
			cv::Mat averaging=cv::Mat::ones(intensityChangeKernelSize, intensityChangeKernelSize, CV_64FC1)/(double)(intensityChangeKernelSize*intensityChangeKernelSize);
			Filter64F(illuminant, illuminant, intensityChangeKernelSize);
		} else{
			GaussianBlur(illuminant, illuminant, cv::Size(intensityChangeKernelSize, intensityChangeKernelSize), intensityChangeSigma);
		}
	}

	std::vector<cv::Mat> destinationChannels;
	split(source, destinationChannels);
	split(illuminant, illuminantChannels);
	for (int i=0;i<(int)destinationChannels.size();++i){
		destinationChannels[i].convertTo(destinationChannels[i], CV_64FC1);
		cv::divide(destinationChannels[i], illuminantChannels[i], destinationChannels[i]);
	}
	
	cv::merge(destinationChannels, destination);
	
	double *check=(double *)destination.data;
	for (int i=0;i<destination.rows*destination.cols*3;++i){
		if (check[i]>=upperBound){
			check[i]=upperBound-1;
		}
	}
	
	destination.convertTo(destination, source.type());

}

int main(int argc, char **argv){

	if (argc<4){
		printf("Usage: %s input_file output_reflectance output_shading [N [n [k1 [k2 [r [c [upper_bound]]]]]]]\n", argv[0]);
		printf("\tN           - number of sprays\n");
		printf("\tn           - size of individual spray\n");
		printf("\tk1          - input kernel size\n");
		printf("\tk2          - intensity change kernel size\n");
		printf("\tr           - rows step\n");
		printf("\tc           - columns step\n");
		printf("\tupper_bound - maximal value for of a pixel channel\n");
		printf("\tnormalize	  - normalize the intensity change (0 for false, 1 for true)\n\n");
		return 0;
	}

	int N=1;
	int n=250;
	int inputKernelSize=25;
	int intensityChangeKernelSize=25;
	int r=1;
	int c=1;
	double upperBound=255.0;
	bool normalizeIntensityChange=false;
	
	cv::Mat img=cv::imread(argv[1], CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);
	
	if (img.rows*img.cols==0){
		return 0;
	}

	if (argc>4){
		sscanf(argv[4], "%d", &N);
		if (argc>5){
			sscanf(argv[5], "%d", &n);
			if (argc>6){
				sscanf(argv[6], "%d", &inputKernelSize);
				if (argc>7){
					sscanf(argv[7], "%d", &intensityChangeKernelSize);
					if (argc>8){
						sscanf(argv[8], "%d", &r);
						if (argc>9){
							sscanf(argv[9], "%d", &c);
							if (argc>10){
								sscanf(argv[10], "%lf", &upperBound);
								if (argc>11){
									int value;
									sscanf(argv[11], "%d", &value);
									normalizeIntensityChange=(bool)value;
								}
							} else if (img.depth()==2){
								upperBound=65535.0;
							}
						}
					}
				}
			}
		}
	}
	
	cv::Mat result;
	PerformRandomSpraysRetinex(img, result, N, n, upperBound, r, c);
	
	result.convertTo(result, CV_8UC3);
	result.convertTo(result, CV_64FC3);
	imwrite(argv[2], result);
    printf("Wrote %s ...\n", argv[2]);
    
    // Compute shading.
    cv::Mat shading(result.rows, result.cols, CV_64FC3);
    cv::Mat img_float;
	img.convertTo(img_float, CV_64FC3);
	
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                shading.at<cv::Vec3d>(i ,j)[c] = std::max(255. - img_float.at<cv::Vec3d>(i ,j)[c]/(result.at<cv::Vec3d>(i, j)[c] + 1)*255., 0.);
            }
        }
    }
    
    shading.convertTo(shading, CV_8UC3);
    cv::cvtColor(shading, shading, CV_BGR2GRAY);
    imwrite(argv[3], shading);
    printf("Wrote %s ...\n", argv[3]);
    
	return 0;
}
