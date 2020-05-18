/**

Dogs and Cats Image Recognition Neural Netowork:

Creating a neural network to identify dogs and cats. 
First time experience with neural networks and image
recognition.

Final Project

Nick Oka & Henzi Kou

CIS 330: C/C++
Professor Jee Whan Choi

Spring 2019

Credits:

Code sourced from OpenNN, an open source neural network library
and Abner M. C. Araujo, an independent author/coder. Roberto Lopez
wrote code on many different use cases within the NN. Araujo helped 
us on how to use openCV and how to read in many files for training and testing.

OpenNN's website:
http://www.opennn.net

Araujo's website:
https://picoledelimao.github.io

Code was pulled from both sources and altered to fit our team's wants in
training our network with dogs and cats and testing based on different test
photos that can be manually entered in by the user; this is completed one 
photo at a time.

Usage:

1. Compile

	make

	or

	g++ dogs_cats_NN.cpp -std=c++0x  -I/usr/local/include/opencv 
	-I/usr/local/include -L/usr/local/lib -lopencv_shape -lopencv_stitching 
	-lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d 
	-lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs 
	-lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann 
	-lopencv_core -o run

2. Run

	./run <TRAINING_IMAGE_DIRECTORY> <TRAINING PERCENTAGE> <TRAINING_PERCENTAGE> 
			<TESTING METHOD> <NETWORK_INPUT_LAYER_SIZE>
	
	1. <TRAINING_IMAGE_DIRECTORY> -> Directory of the training images.
	2. <TRAINING_PERCENTAGE> -> Percent of training directory to be trained (0,1.00]
	3. <TESTING METHOD> -> 1 -> Equal percentage training/testing (0, 0.50]
						   2 -> Train/Test Ratio. Will read tthe entire train directory
						   3 -> Individual image test
	4. <NETWORK_INPUT_LAYER_SIZE> -> Default is 510. Input to use a different value.

**/

#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>
#include <dirent.h>
#include <sys/types.h>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

// to iterate through file vector
typedef std::vector<std::string>::const_iterator vec_iter;

struct ImageData
{
	std::string classname;
	cv::Mat bowFeatures;
};

/**

Gets all files from the given directory and stores into a vector.
The parameter is the name of the directory in which the files are
contained.
Will return the list the of files in the form of a vector.

**/
std::vector<std::string> 
getFilesinDirectory(const std::string& directory)
{
	std::vector <std::string> files;

	DIR *dirp = opendir(directory.c_str());
	struct dirent *dp;

	while((dp = readdir(dirp)) != NULL)
	{
		files.push_back(dp -> d_name);
	}

	closedir(dirp);

	// add the pathway to the filename in the vector
	for (int i = 0; i < files.size(); i++)
	{
		std::string str = files[i];
		files[i] = directory + str;
	}

	return files;
}

/**

Gets the class name of the given filename.
The parameter is the name of the file.
Will return the class name as a string.

**/
inline std::string
getClassName(const std::string& filename)
{
	return filename.substr(filename.find_last_of('/') + 1, 3);
}

/**

Gets the "descriptors" of the given image. The "descriptors" 
uses the KAZE algorithm to get the KAZE features of an image. 
Sets up for the BOW function.
The parameter is the image.
Will return the "descriptors."

**/
cv::Mat
getDescriptors(const cv::Mat& img)
{
	cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

	return descriptors;
}

/**

Iterates through the given vector and reads each image from the names
of the files that are in the vector. 

Verifies if the fiilenames are valid and can be read. 

**/
void
readImages(vec_iter begin, vec_iter end,
	std::function<void (const std::string&, const cv::Mat&)> callback)
{
	for (auto it = begin; it != end; ++it)
	{
		std::string filename = *it;
		std::cout << "Reading file as an image: " << filename << std::endl;
		cv::Mat img = cv::imread(filename);

		if (img.empty())
		{
			std::cerr << "Invalid Image File: Could not read image." << std::endl;
			continue;
		}

		std::string classname = getClassName(filename);
		cv::Mat descriptors = getDescriptors(img);
		callback(classname, descriptors);
	}
}

/**

Gets the id from the classname and used for getting the binary
code of a class. 

**/
int getClassId(const std::set<std::string>& classes, const std::string& classname)
{
	int index = 0;
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		if (*it == classname)
		{
			break;
		}
		++index;
	}
	return index;
}

/**

Gets the binary code associated to a class for use when preparing
the neural network.

**/
cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname)
{
	cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
	int index = getClassId(classes, classname);
	code.at<float>(index) = 1;
	return code;
}

/**

Turn local features into a single bag of words histogram of 
of visual words (a.k.a., bag of words features)

**/
cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors,
  int vocabularySize)
{
	cv::Mat outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
	std::vector<cv::DMatch> matches;
	flann.match(descriptors, matches);
	for (size_t j = 0; j < matches.size(); j++)
	{
		int visualWord = matches[j].trainIdx;
		outputArray.at<float>(visualWord)++;
	}

	return outputArray;
}

/**

Get a trained neural network.
 
**/
cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples,
  const cv::Mat& trainResponses)
{
	int networkInputSize = trainSamples.cols;
	int networkOutputSize = trainResponses.cols;
	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
	std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2,
	  networkOutputSize };
	mlp->setLayerSizes(layerSizes);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
	return mlp;
}

/**

Receives the image that is being tested and gets the index
with the highest probability chance.
Returns the id with the highest probability.

**/
int getPredictedClass(const cv::Mat& predictions)
{
	float maxPrediction = predictions.at<float>(0);
	float maxPredictionIndex = 0;
	const float* ptrPredictions = predictions.ptr<float>(0);
	for (int i = 0; i < predictions.cols; i++)
	{
		float prediction = *ptrPredictions++;
		if (prediction > maxPrediction)
		{
			maxPrediction = prediction;
			maxPredictionIndex = i;
		}
	}

	return maxPredictionIndex;
}

/**

Gets the id from prediction of the trained neural network.
The id returned corresponds to either a dog or a cat.

0 - dog
1 - cat
 
**/
int getDogCat(cv::Ptr<cv::ml::ANN_MLP> mlp,
  const cv::Mat& testSamples, int counter)
{
	cv::Mat testOutput;
	// predicts using the test photo vector
	mlp->predict(testSamples, testOutput);

	// id will be either 0 or 1 depicting dog or cat
	int id = getPredictedClass(testOutput.row(counter));
	return id;
}

/**

Won't work on ix-dev

Displays the image given the file pathway.
Will check if the pathway whether it is a
valid image or not.

Displays the text of if it's a dog or a cat

**/
void
displayImage(std::string result, std::string file)
{

	cv::Mat testImg = cv::imread(file);

	if (!testImg.empty())
	{
		// read an image
	    cv::Mat image= cv::imread(file);   
	    // create image window named "dog or cat"
	    cv::namedWindow(result);
	    // show the image on window
	    cv::imshow(result, image);
	    // wait key for 50 ms
	    cv::waitKey(50);
	    // close image window
	    cv::destroyAllWindows();
	}

}

/**

calculates the accuracy of the test directory and prints
the expected vs predicted amounts that are used in the
calculation of the accuracy.

**/
float
accuracyCalculation(int actual_dog, int actual_cat, int guess_dog,
	int guess_cat)
{
	// int correct_dogs = std::min(actual_dog, guess_dog);
 //  	int correct_cats = std::min(actual_cat, guess_cat);
 //  	int not_correct_dogs = std::max(actual_dog, guess_dog) - correct_dogs;
 //  	int not_correct_cats = std::max(actual_cat, guess_cat) - correct_cats;

  	// std::cout << not_correct_cats << " " <<  not_correct_dogs << std::endl;

  	std::cout << std::endl;
  	std::cout << "Expected dogs: " << actual_dog << std::endl;
  	std::cout << "Predicted dogs: " << guess_dog << std::endl;
  	std::cout << "Expected cats: " << actual_cat << std::endl;
  	std::cout << "Predicted cats: " << guess_cat << std::endl;

  	// float add_correct = correct_cats + correct_dogs;
  	// float add_incorrect = not_correct_dogs + not_correct_cats;
  	// float total = add_correct + add_incorrect;

  	float add_correct = guess_cat + guess_dog;
  	// float add_incorrect = not_correct_dogs + not_correct_cats;
  	float total = actual_cat + actual_dog;

  	// std::cout << add_correct << " " << total <<  std::endl;

  	return add_correct / total;
}

int
main(int argc, char** argv)
{
	if (argc < 4 || argc > 5)
	{
		std::cerr << "Usage: <TRAINING_IMAGE_DIRECTORY> " 
				"<TRAINING PERCENTAGE> <TRAINING_PERCENTAGE>" 
				"<TESTING METHOD> <NETWORK_INPUT_LAYER_SIZE>"<< std::endl;
		std::cerr << "1. <TRAINING_IMAGE_DIRECTORY> -> "
			"Directory of the training images." << std::endl;
		std::cerr << "2. <TRAINING_PERCENTAGE> -> "
			"Percent of training directory to be trained (0,1.00]" << std::endl;
		std::cerr << "3. <TESTING METHOD> -> "
			"1 -> Equal percentage training/testing (0, 0.50]" << std::endl;
			std::cerr << "		       2 -> Train/Test Ratio. Will read tthe entire "
			"train directory" << std::endl;
			std::cerr << "		       3 -> Individual image test" << std::endl;
		std::cerr << "4. <NETWORK_INPUT_LAYER_SIZE> -> "
			"Default is 510. Input to use a different value" << std::endl;
		exit(-1);
	}

	std::string imagesDir = argv[1];
	float trainSplitRatio = atof(argv[2]);
	int test_method = atoi(argv[3]);
	int networkInputSize = 510;
	if(test_method < 1 || test_method > 3)
	{
		std::cerr << "Testing method must be between 1, 2, or 3." << std:: endl;
		std::cerr << "1 -> Equal percentage training/testing (0, 0.50]" << std::endl;
			std::cerr << "		       2 -> Train/Test Ratio. Will read tthe entire "
			"train directory" << std::endl;
			std::cerr << "		       3 -> Individual image test" << std::endl;
		std::cerr << std::endl;
		exit(-1);
	}
	if (argc == 5)
	{
		networkInputSize = atoi(argv[4]);
	}

	if (test_method == 1 && trainSplitRatio > 0.50)
	{
		std::cerr << "Training Percentage must be a float (0, 0.50]" << std:: endl;
		std::cerr << std::endl;
		exit(-1);
	}

	if (trainSplitRatio <= 0.00 || trainSplitRatio > 1.00)
	{
		if (test_method == 1)
		{
			std::cerr << "Training Percentage must be a float (0, 0.50]" << std:: endl;
			std::cerr << std::endl;
			exit(-1);
		}
		else
		{
			std::cerr << "Training Percentage must be a float (0, 1.00]" << std:: endl;
			std::cerr << std::endl;
			exit(-1);
		}
	}

	// too small of a networkInputSize results in aborts
	if (networkInputSize <= 9)
	{
		std::cerr << "Network Input Layer must be an int > 9" << std:: endl;
		std::cerr << std::endl;
		exit(-1);
	}

	// an abort error occurred when testing with the entire folder
	// .99 of the training folder is used in this case
	if (trainSplitRatio == 1)
	{
		trainSplitRatio = .99;
	}

	// reading all of the images from the training directory
	std::cout << "Reading the training set now!" << std::endl;
	double start = (double)cv::getTickCount(); // start timer
	std::vector<std::string> files = getFilesinDirectory(imagesDir);
	std::random_shuffle(files.begin(), files.end()); // shuffle the list of files

	cv::Mat descriptorsSet;
	std::vector<ImageData*> descriptorsMetadata;
	std::set<std::string> classes;
	readImages(files.begin(), files.begin() + (size_t)(files.size() * trainSplitRatio),
		[&](const std::string& classname, const cv::Mat& descriptors) 
	{
	    // Append to the set of classes
	    classes.insert(classname);
	    // Append to the list of descriptors
	    descriptorsSet.push_back(descriptors);
	    // Append metadata to each extracted feature
	    ImageData* data = new ImageData;
	    data->classname = classname;
	    data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
	    for (int j = 0; j < descriptors.rows; j++)
	    {
	        descriptorsMetadata.push_back(data);
	    }
  	});

  	double count = ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0;

	if (count > 1.00)
	{
		std::cout << "Time elapsed in minutes: " << count << std::endl;
	}

	std::cout << "Creating vocabulary now!" << std::endl;
	start = (double)cv::getTickCount();
	cv::Mat labels;
	cv::Mat vocabulary;
	// Use k-means to find k centroids (the words of our vocabulary)
	cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS +
      	cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
	// Don't need to keep it memory
	descriptorsSet.release();
	count = ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0;

	// will only display the time if it's over a minute
	if (count > 1.00)
	{
		std::cout << "Time elapsed in minutes: " << count << std::endl;
	}

  	// Convert a set of local features for each image in a single descriptors
	// using the bag of words technique
	std::cout << "Getting histograms of visual words now!" << std::endl;
	int* ptrLabels = (int*)(labels.data);
	int size = labels.rows * labels.cols;
	for (int i = 0; i < size; i++)
	{
		int label = *ptrLabels++;
		ImageData* data = descriptorsMetadata[i];
		data->bowFeatures.at<float>(label)++;
	}

	// Filling matrices to be used by the neural network
	std::cout << "Preparing neural network now!" << std::endl;
	cv::Mat trainSamples;
	cv::Mat trainResponses;
	std::set<ImageData*> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());

	for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end(); )
	{
		ImageData* data = *it;
		cv::Mat normalizedHist;
		cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
		trainSamples.push_back(normalizedHist);
		trainResponses.push_back(getClassCode(classes, data->classname));
		delete *it; // clear memory
		it++;
	}

	descriptorsMetadata.clear();
  
	// Training the neural network
	std::cout << "Training the neural network now!" << std::endl;
	start = cv::getTickCount();
	cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
	count = ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0;

	if (count > 1.00)
	{
		std::cout << "Time elapsed in minutes: " << count << std::endl;
	}
  
	// memory can be freed
	trainSamples.release();
	trainResponses.release();
  
	// Train FLANN - Functional link artificial neural network
	std::cout << "Training FLANN (Functional link artificial neural network)!" << std::endl;
	start = cv::getTickCount(); // start timer
	cv::FlannBasedMatcher flann;
	flann.add(vocabulary);
	flann.train();
	count = ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0;

	if (count > 1.00)
	{
		std::cout << "Time elapsed in minutes: " << count << std::endl;
	}

	if (test_method == 1)
	{

		std::cout << "Reading the test set now!" << std::endl;
		start = (double)cv::getTickCount(); // start timer
		// std::vector<std::string> test_files = getFilesinDirectory(imagesDir);
		// std::random_shuffle(test_files.begin(), test_files.end()); // shuffle the list of files

		int counter = (size_t)(files.size() * trainSplitRatio);
		int here = 0;
		int actual_dog = 0;
		int actual_cat = 0;
		int guess_dog = 0;
		int guess_cat = 0;

		cv::Mat testSamples;

		readImages(files.begin() + (size_t)(files.size() * trainSplitRatio), 
			files.begin() + (size_t)(files.size() * trainSplitRatio) + 
			+ (size_t)(files.size() * trainSplitRatio),
	      [&](const std::string& classname, const cv::Mat& descriptors)
		{

			std::string cur_file = files[counter];

			std::vector <std::string> test_photo_max; // vector for the test photo
			test_photo_max.push_back(cur_file);

			// Get histogram of visual words using bag of words technique
			cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
			cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
			testSamples.push_back(bowFeatures);

			std::string test_classname = getClassName(cur_file);

			if (test_classname == "dog")
			{
			    actual_dog += 1;
			}
			else if (test_classname == "cat")
			{
			    actual_cat += 1;
			}

			int id = getDogCat(mlp, testSamples, here);

			// std::cout << id << std::endl;

			if(test_classname == "dog" && id == 1)
			{
				guess_dog += 1;
				// std::cout << guess_dog << std::endl;
			}
			else if (test_classname == "cat" && id == 0)
			{
				guess_cat += 1;
				// std::cout << guess_cat << std::endl;
			}

		    // std::cout << files[counter] << std::endl;
		    here ++;
		    counter ++;
	  	});

	  	count = ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0;

		if (count > 1.00)
		{
			std::cout << "Time elapsed in minutes: " << count << std::endl;
		}

	  	float acc_perc = accuracyCalculation(actual_dog, actual_cat,
	  		guess_dog, guess_cat);

	  	std::cout << "Accuracy: " <<  acc_perc * 100 << "%" << std::endl;

	}

	else if (test_method == 2)
	{

		std::cout << "Reading the test set now!" << std::endl;
		start = (double)cv::getTickCount(); // start timer
		// std::vector<std::string> test_files = getFilesinDirectory(imagesDir);
		// std::random_shuffle(test_files.begin(), test_files.end()); // shuffle the list of files

		int counter = (size_t)(files.size() * trainSplitRatio);
		int here = 0;
		int actual_dog = 0;
		int actual_cat = 0;
		int guess_dog = 0;
		int guess_cat = 0;

		cv::Mat testSamples;

		readImages(files.begin() + (size_t)(files.size() * trainSplitRatio), files.end(),
	      [&](const std::string& classname, const cv::Mat& descriptors)
		{

			std::string cur_file = files[counter];

			std::vector <std::string> test_photo_max; // vector for the test photo
			test_photo_max.push_back(cur_file);

			// Get histogram of visual words using bag of words technique
			cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
			cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
			testSamples.push_back(bowFeatures);

			std::string test_classname = getClassName(cur_file);

			if (test_classname == "dog")
			{
			    actual_dog += 1;
			}
			else if (test_classname == "cat")
			{
			    actual_cat += 1;
			}

			int id = getDogCat(mlp, testSamples, here);

			// std::cout << id << std::endl;

			if(test_classname == "dog" && id == 1)
			{
				guess_dog += 1;
				// std::cout << guess_dog << std::endl;
			}
			else if (test_classname == "cat" && id == 0)
			{
				guess_cat += 1;
				// std::cout << guess_cat << std::endl;
			}

		    // std::cout << files[counter] << std::endl;
		    here ++;
		    counter ++;
	  	});

	  	count = ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0;

		if (count > 1.00)
		{
			std::cout << "Time elapsed in minutes: " << count << std::endl;
		}

	  	float acc_perc = accuracyCalculation(actual_dog, actual_cat,
	  		guess_dog, guess_cat);

	  	std::cout << "Accuracy: " <<  acc_perc * 100 << "%" << std::endl;

	}

	else{

		// Testing process of images - one photo at a time
		int a = 0;
		while (a == 0){
			std::string file;
			std::cout << "Input pathway of test photo ('done' when done): " << std::endl;
			std::cin >> file;
			cv::Mat testImg = cv::imread(file, 0);

			// done testing images
			if (file == "done")
			{
				a = 1;
			}
			// if the pathway inputted isnt a readable image
			else if (testImg.empty())
			{
				std::cout << "Invalid Image File: Could not read image." << std::endl;
			}
			else
			{
				std::cout << "Reading the test image file now!" << std::endl;
				start = cv::getTickCount(); // starting timer
				cv::Mat testSamples;
				std::vector <std::string> test_photo; // vector for the test photo
				test_photo.push_back(file);

				readImages(test_photo.begin() + (size_t)(test_photo.size() * trainSplitRatio), test_photo.end(),
					[&](const std::string& classname, const cv::Mat& descriptors) 
				{
					// Get histogram of visual words using bag of words technique
					cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
					cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
					testSamples.push_back(bowFeatures);

				});

				count = ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0;

				if (count > 1.00)
				{
					std::cout << "Time elapsed in minutes: " << count << std::endl;
				}

				std::cout << std::endl;

				// Get id of the test set
				int id = getDogCat(mlp, testSamples, 0);

				std::string result; // store result for image window

				// 1 -> dog
				if (id == 1)
				{
					std::cout << "It's a DOG!" << std::endl;
					result = "A DOG!";
				}
				// 0 -> cat
				else
				{
					std::cout << "It's a CAT" << std::endl;
					result = "A CAT!";
				}

				displayImage(result, file);

			}
		}

	}

	// end of program
	std::cout << std::endl;
	std::cout << "Closing NN!" << std::endl;
	std::cout << "End of program!" << std::endl;

	return 0;

}
