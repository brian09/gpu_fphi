#include<cmath>
#include<stdio.h>
#include<vector>
#include<algorithm>
#include<sstream>
#include<fstream>
#include<iostream>
#include<string>
#include <time.h>
#include<random>




static bool return_isspace (char  x){
	if(std::isspace(x)){
		return true;
	}
	return false;
}

static std::vector<float> parse_line(std::string currentLine){

	currentLine.erase(std::remove_if(currentLine.begin(), currentLine.end(), return_isspace), currentLine.end());
	std::vector<float> output_vector;
	std::string current_str;
	char current_letter;

	for(std::string::iterator it = currentLine.begin(); it != currentLine.end(); it++){
		current_letter = *it;
		if(current_letter == ','){
			output_vector.push_back(std::atof(current_str.c_str()));
			current_str.clear();
		}else{
			current_str += current_letter;
		}
	}
	output_vector.push_back(std::atof(current_str.c_str()));


	return output_vector;

}



static float * csvread_float(const char * filename, size_t &rows,
	    size_t &columns) {


	std::ifstream inputData(filename);
;
	if(inputData.is_open() == false){
		std::cerr << "Error in opening file: " << filename << "\n";
		return NULL;
	}
	std::string Line;

	std::vector< std::vector<float> > data;

	while (std::getline(inputData, Line)) {

		data.push_back(parse_line(Line));

	}

	rows = data.size();

	columns = data[0].size();

	inputData.close();

	float * data_out = new float [rows*columns];


	size_t colIdx = 0;
	size_t rowIdx = 0;
	for(std::vector< std::vector<float> >::iterator iter = data.begin(); iter != data.end() ; iter++, rowIdx++){
		colIdx = 0;
		for(std::vector<float>::iterator current_value = iter->begin() ; current_value != iter->end(); current_value++, colIdx++){
			data_out[colIdx*rows + rowIdx] = *current_value;
		}
	}


	return data_out;

}



static unsigned int * create_permutation_matrix(unsigned int n_subjects, unsigned int n_permutations) {




  unsigned int * pmatrix = new unsigned int[n_subjects * n_permutations];
  srand(0);
	for(unsigned int j = 0 ; j < n_permutations; j++){


		std::vector<unsigned int> randNums;

		for (int r = 0; r < n_subjects; r++) {
			randNums.push_back(r);
		}
		std::random_device rd;
		std::mt19937 mt(rd());

	    for (unsigned int n = n_subjects; n > 0; n--) {

	        std::uniform_int_distribution<unsigned int> dist(0.0,  n - 1);
	    	unsigned int Idx = dist(mt);
			pmatrix[(n - 1) + j*n_subjects] = randNums[Idx];
			randNums.erase(randNums.begin() + Idx);
	    }






	}



	return pmatrix;

}

static std::string convert_float_to_string(float value){
  std::stringstream ss;
  ss << value;
  std::string str_value(ss.str());

  return str_value;
}
static void write_to_file(std::string file_name, std::string header_file, float * h2, float * indicator, float * pval, bool get_pvalue, size_t n_voxels){

  std::ifstream file_in(header_file.c_str());
  std::vector< std::string > lines;

  if(file_in.is_open()){

	  for(size_t voxel = 0; voxel < n_voxels; voxel++){
		  std::string line;
		  if(file_in.eof()){
			  std::cout << "Warning the header file does not have enough trait names to correspond to the trait matrix column number.\n";


			  std::cout << "Proceeding anyway with trait number " << voxel + 1 << " out of " << n_voxels << "traits.\n";

			  line = ',' +  convert_float_to_string(indicator[voxel]) + ',' + convert_float_to_string(h2[voxel]);
			  if(get_pvalue){
				  line +=  ',' + convert_float_to_string(pval[voxel]) + '\n';
			  }else{
				  line += '\n';
			  }


		  }else{
			  std::getline(file_in,line);

		  	  line += ',' +  convert_float_to_string(indicator[voxel]) + ',' + convert_float_to_string(h2[voxel]);
		  	  if(get_pvalue){
		  		  line +=  ',' + convert_float_to_string(pval[voxel]) + '\n';
		  	  }else{
		  		  line += '\n';
		  	  }

		  }

		  lines.push_back(line);
	  }

  }else{
	  std::cout << "Warning header file was not found.  Proceeding anyway without trait names.\n";

	  for(size_t voxel = 0; voxel < n_voxels; voxel++){
		  std::string line;
		  line = "N/A,";
		  line +=  convert_float_to_string(indicator[voxel]) + ',' + convert_float_to_string(h2[voxel]);
		  if(get_pvalue){
			  line +=  ',' + convert_float_to_string(pval[voxel]) + '\n';
		  }else{
			  line += '\n';
		  }
		  lines.push_back(line);
	  }
  }

  file_in.close();

  std::ofstream file_out(file_name.c_str());
  file_out << "Trait,Indicator,H2r";
  if(get_pvalue)
	  file_out << ",pvalue\n";
  else
	  file_out << "\n";

  for(std::vector< std::string >::iterator it = lines.begin() ; it != lines.end() ; it++){

	  file_out << *it;

  }

  file_out.close();

}

extern "C" std::vector<int> select_devices();
extern "C" int call_cudafphi(float * h2, float * indicator, float * pvals,float * h_y,float * h_Z,float * h_hat,
    float * h_evectors, unsigned int * h_pmatrix, bool covariates, bool get_pval,
    size_t n_voxels, size_t n_subjects, size_t n_permutations, std::vector<int> selected_devices);


void print_help(){
	printf("cuda_fphi calculates the heritability and indicator values of a set of traits that share an auxiliary and eigenvector matrix.\n");
	printf("Optionally the pvalue can be calculated given a number of permutations and covariates can be included given a hat matrix.\n");
	printf("Where hat = I - X * (XT * X)^-1 * XT\n");
	printf("cuda_fphi <trait matrix file name> <auxiliary matrix filename> <eigenvector matrix (non-transposed) filename>  <output file name> <trait header file name> optional args: <n_permutations> <hat matrix file name>\n");
}

int main (int argc, const char *argv[]){



	if(argc != 6 && argc != 7 && argc != 8){
		print_help();
	  	return EXIT_FAILURE;
	}


	std::vector<int> selected_devices = select_devices();

	if(selected_devices.size() == 0){
		printf("No usable devices were selected.\n");
		return EXIT_FAILURE;
	}



	size_t  n_subjects;

	size_t  n_permutations = -1;
	size_t  n_voxels;
	float * h_y;
	try{
		h_y = csvread_float(argv[1], n_subjects, n_voxels);


	}catch(std::bad_alloc&){
		std::cout << "Failed to allocate the memory needed for the trait matrix.\n";
		return EXIT_FAILURE;
	}catch(...){
		std::cout << "Unknown failure in trying to load the trait matrix.\n";
		return EXIT_FAILURE;
	}

 	if(h_y == NULL){
 		return EXIT_FAILURE;
 	}

 	size_t n_subjects_1;
 	size_t n_subjects_2;
 	float * evectors;


	try{
		evectors = csvread_float(argv[3], n_subjects_1, n_subjects_2);
	}catch(std::bad_alloc&){
		std::cout << "Failed to allocate the memory needed for the eigenvector matrix.\n";
		return EXIT_FAILURE;
	}catch(...){
		std::cout << "Unknown failure in trying to load the eigenvector matrix.\n";
		return EXIT_FAILURE;
	}

 	if(evectors == NULL){
 		delete [] h_y;
 		return EXIT_FAILURE;
 	}

 	if(n_subjects_1 != n_subjects_2){
 		printf("Row size and column size are not equal for evector matrix.\n There are %i rows and %i columns.\n",
			  n_subjects_1, n_subjects_2);
 		delete [] h_y;
 		delete [] evectors;
 		return EXIT_FAILURE;
 	}

 	if(n_subjects_1 != n_subjects){
 		printf("Rows of trait matrix and eigenvector matrix are not equal.\n %i rows in the trait matrix and %i rows and columns in the eigenvector matrix.\n",
			  n_subjects, n_subjects_1);
 		delete [] h_y;
 		delete [] evectors;
 		return EXIT_FAILURE;
 	}

 	size_t  dummy;
 	float * hat;
 	float *  aux;
	try{
		aux = csvread_float(argv[2], n_subjects_1, dummy);
	}catch(std::bad_alloc&){
		std::cout << "Failed to allocate the memory needed for the auxiliary matrix.\n";
		return EXIT_FAILURE;
	}catch(...){
		std::cout << "Unknown failure in trying to load the auxiliary matrix.\n";
		return EXIT_FAILURE;
	}


 	if(aux == NULL){
 		delete [] evectors;
 		delete [] h_y;
 		return EXIT_FAILURE;
 	}

 	if(dummy != 2){
 		printf("Auxiliary matrix requires only two columns.  %i columns were detected.\n", dummy);
 		delete [] aux;
 		delete [] evectors;
 		delete [] h_y;
 		return EXIT_FAILURE;
 	}

 	if(n_subjects_1 != n_subjects){
 		printf("Auxiliary matrix subject number is %i while the trait matrix subject number is %i.\n",n_subjects_1, n_subjects);
 		delete [] aux;
 		delete [] evectors;
 		delete [] h_y;
 		return EXIT_FAILURE;
 	}


 	bool covariates = false;
 	bool get_pval = false;
 	unsigned int * pmatrix;


 	if(argc == 7){

 		try{

 			n_permutations = std::strtol(argv[6], NULL, 10);

 			if(n_permutations <= 0)
 				throw 0;
 			get_pval = true;
 			try{
 				pmatrix = create_permutation_matrix(n_subjects, n_permutations);
 			}catch(std::bad_alloc&){
 				std::cout << "Could not allocate memory for permutation matrix.\n";
 				return EXIT_FAILURE;
 			}catch(...){
 				std::cout << "Unknown failure in trying to create permutation matrix.\n";
 				return EXIT_FAILURE;
 			}
 		}
 		catch(...){

 			try{
 	 			hat = csvread_float(argv[6], n_subjects_1, n_subjects_2);
 			}catch(std::bad_alloc&){
 				std::cout << "Could not allocate memory for hat matrix.\n";
 				return EXIT_FAILURE;
 			}catch(...){
 				std::cout << "Unknown failure in trying to create hat matrix.\n";
 				return EXIT_FAILURE;
 			}
 			covariates = true;
 			if(hat == NULL){
 				printf("Error in argument 6: %s\n", argv[6]);
 				delete [] aux;
 				delete [] evectors;
 				delete [] h_y;
 				return EXIT_FAILURE;
 			}

 			if(n_subjects_1 != n_subjects_2){
 				printf("Hat matrix rows and columns are not equal.\n  It has %i row and %i columns.\n", n_subjects_1, n_subjects_2);
 				delete [] aux;
 				delete [] evectors;
 				delete [] h_y;
 				delete [] hat;
 				return EXIT_FAILURE;
 			}

 			if(n_subjects_1 != n_subjects){
 				printf("Rows of trait matrix are not equal to rows of hat matrix.\n  The number of rows in the trait matrix is %i and the number of rows in the hat matrix is %i.\n",
 						n_subjects, n_subjects_1);
 				delete [] aux;
 				delete [] evectors;
 				delete [] h_y;
 				delete [] hat;
 				return EXIT_FAILURE;
 			}

 			covariates = true;
 		}
 	}else if (argc == 8){
 		n_permutations = std::strtol(argv[6], NULL, 10);
 		if(n_permutations <= 0){
 			printf("An invalid number was entered for n_permutations: %s\n", argv[5]);
			delete [] aux;
			delete [] evectors;
			delete [] h_y;
			return EXIT_FAILURE;
 		}
 		get_pval = true;
 		try{
 			pmatrix = create_permutation_matrix(n_subjects, n_permutations);
 		}catch(std::bad_alloc&){
 			std::cout << "Could not allocate memory for permutation matrix.\n";
			return EXIT_FAILURE;
		}catch(...){
			std::cout << "Unknown failure in trying to create permutation matrix.\n";
			return EXIT_FAILURE;
		}
		try{
			hat = csvread_float(argv[7], n_subjects_1, n_subjects_2);
		}catch(std::bad_alloc&){
			std::cout << "Could not allocate memory for hat matrix.\n";
			return EXIT_FAILURE;
		}catch(...){
			std::cout << "Unknown failure in trying to create hat matrix.\n";
			return EXIT_FAILURE;
		}

 		covariates = true;
 		if(hat == NULL){
 			delete [] pmatrix;
 			delete [] aux;
 			delete [] h_y;
 			delete [] evectors;
 			return EXIT_FAILURE;
 		}

 		if(n_subjects_1 != n_subjects_2){
 			printf("Hat matrix rows and columns are not equal.\n  It has %i row and %i columns.\n", n_subjects_1, n_subjects_2);
			delete [] aux;
			delete [] evectors;
			delete [] h_y;
			delete [] hat;
			delete [] pmatrix;
			return EXIT_FAILURE;
		}

		if(n_subjects_1 != n_subjects){
			printf("Rows of trait matrix are not equal to rows of hat matrix.\n  The number of rows in the trait matrix is %i and the number of rows in the hat matrix is %i.\n",
					n_subjects, n_subjects_1);
			delete [] aux;
			delete [] evectors;
			delete [] h_y;
			delete [] hat;
			delete [] pmatrix;
			return EXIT_FAILURE;
		}

 	}

 	std::string output_filename(argv[4]);

 	float * indicator = new float[n_voxels];
 	float * pvals = new float[n_voxels];
 	float * h2 = new float[n_voxels];
 	call_cudafphi(h2, indicator, pvals, h_y, aux, hat, evectors, pmatrix, covariates, get_pval, n_voxels, n_subjects, n_permutations, selected_devices);
 	write_to_file(output_filename,std::string(argv[5]), h2, indicator, pvals, get_pval, n_voxels);
 	delete [] h2;
 	delete [] pvals;
 	delete [] indicator;
 	delete [] aux;
 	delete [] evectors;
 	delete [] h_y;
 	if(covariates)
 		delete [] hat;
 	if(get_pval)
 		delete [] pmatrix;


 	return EXIT_SUCCESS;
}




