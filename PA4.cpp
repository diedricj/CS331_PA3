// PA4.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <iterator>


struct x {
    int pos;
    int neg;
    double p_pos;
    double p_neg;

    x() {
        pos = 0;
        neg = 0;
    }
};

void clear_punc(std::vector<std::string>& sentences) {
    for (int i = 0; i < sentences.size(); i++) {
        for (int j = 0, len = sentences[i].size(); j < len; j++) {
            if (std::ispunct((unsigned char)sentences[i][j]))
            {
                sentences[i].erase(j--, 1);

                len = sentences[i].size();
            }
        }
    }
}

std::vector<std::string> create_vocab(std::vector<std::vector<std::string>> data) {
    std::string word;
    std::vector<std::string> vocab;
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
            vocab.push_back(data[i][j]);
        }
    }
    return vocab;
}

void lower(std::vector<std::string> &sentences) {
    for (int i = 0; i < sentences.size(); i++) {
        std::transform(sentences[i].begin(), sentences[i].end(), sentences[i].begin(),
            [](unsigned char c) { return std::tolower(c); });
    }
}


std::vector<std::vector<int>> generate_feature_vector(std::vector<std::vector<std::string>> data, std::vector<std::string> vocab) {
    std::vector<std::vector<int>> features(data.size());
    char result;
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < vocab.size(); j++) {
            //if data[i] conatains vocab[i]
            if (std::find(data[i].begin(), data[i].end(), vocab[j]) != data[i].end())
            {
                features[i].push_back(1);
            }
            else {
                features[i].push_back(0);
            }
        }
        features[i].push_back(std::stoi(data[i][data[i].size() - 1]));
    }
    return features;
}

std::vector<std::vector<std::string>> itemize(std::vector<std::string> sentences) {
    std::vector<std::vector<std::string>> items;
    for (int i = 0; i < sentences.size(); i++) {
        std::stringstream ss(sentences[i]);
        std::istream_iterator<std::string> begin(ss);
        std::istream_iterator<std::string> end;
        std::vector<std::string> vstrings(begin, end);  //this method of itemizing was found on stack overflow and implemented here
        items.push_back(vstrings);
    }
    return items;

}

std::vector<x> learn_nb(std::vector<std::vector<int>> pre_processed_train, int size, double &p_pos, double &p_neg) {
    std::vector<x> word_counts(size);
    int num_positive_reviews;
    double n_pos = 0;
    double n_neg = 0;

    for (int i = 0; i < pre_processed_train.size(); i++) {
        if (pre_processed_train[i][pre_processed_train[i].size() - 1]) {
            n_pos++;
            for (int j = 0; j < pre_processed_train[i].size()-1; j++) {
                word_counts[j].pos += pre_processed_train[i][j];
            }
        }
        else {
            n_neg++;
            for (int j = 0; j < pre_processed_train[i].size() - 1; j++) {
                word_counts[j].neg += pre_processed_train[i][j];
            }
        }
    }

    for (int i = 0; i < word_counts.size(); i++) {
        word_counts[i].p_pos = (word_counts[i].pos + 1.0) / (n_pos + size);
        word_counts[i].p_neg = (word_counts[i].neg + 1.0) / (n_neg + size);
    }

    p_pos = n_pos / (n_pos + n_neg);
    p_neg = n_neg / (n_pos + n_neg);

    return word_counts;
}

int predict(double p_pos, double p_neg, std::vector<int> data, std::vector<x> word_counts) {
    double ln_p_x_given_true = 0;
    double ln_p_x_given_false = 0;

    for (int i = 0; i < data.size()-1; i++) {
        ln_p_x_given_false += log(word_counts[i].p_neg) * data[i];
        ln_p_x_given_true += log(word_counts[i].p_pos) * data[i];
    }

    double pos_pred = ln_p_x_given_true + log(p_pos);
    double neg_pred = ln_p_x_given_false + log(p_neg);
    if(pos_pred >= neg_pred)
        return 1;
    else
        return 0;
}

double prediction_percent_correct(double p_pos, double p_neg, std::vector<std::vector<int>> data, std::vector<x> word_counts) {
    int total_correct = 0;
    for (int i = 0; i < data.size(); i++) {
        if (predict(p_pos, p_neg, data[i], word_counts) == data[i][data[i].size() - 1]) {
            total_correct++;
        }
    }
    return total_correct / (double)data.size();
}

std::vector<std::string> extract_sentences(std::ifstream &infile) {
    std::vector<std::string> sentences;
    std::string line;
    while (std::getline(infile, line)) {
        sentences.push_back(line);
    }
    infile.close();

    lower(sentences);
    clear_punc(sentences);
    return sentences;
}

void output_featurized_vector(std::ofstream& outfile, std::vector<std::vector<int>> feature_vector, std::vector<std::string> vocab) {
    for(int i = 0; i < vocab.size(); i++){
        outfile << vocab[i] << ",";
    }
    outfile << "classlabel\n";
    for (int i = 0; i < feature_vector.size(); i++) {
        for (int j = 0; j < feature_vector[i].size(); j++) {
            outfile << feature_vector[i][j];
            if (feature_vector[i].size() - 1 != j) {
                outfile << ",";
            }
            else {
                outfile << "\n";
            }
        }
    }
}

int main()
{
    std::ifstream infile;
    infile.open("trainingSet.txt");
    std::vector<std::string> train_sentences = extract_sentences(infile);
    std::vector<std::vector<std::string>> train_data = itemize(train_sentences); //format training input

    std::vector<std::string> vocab;

    vocab = create_vocab(train_data);  //create vocab from testing data
    std::sort(vocab.begin(), vocab.end());  //sort alphabetically
    std::vector<std::string>::iterator ip = std::unique(vocab.begin(), vocab.end());  //remove duplicates
    vocab.resize(std::distance(vocab.begin(), ip)); //remove duplicates

    vocab.erase(std::remove(vocab.begin(), vocab.end(), "0"), vocab.end());  //dont include class label as a word
    vocab.erase(std::remove(vocab.begin(), vocab.end(), "1"), vocab.end()); //dont include class label as a word
    std::vector<std::vector<int>> pre_processed_train = generate_feature_vector(train_data, vocab);  //generate feature vector from training data

    double p_pos;
    double p_neg;
    std::vector<x> word_counts = learn_nb(pre_processed_train, vocab.size(), p_pos, p_neg);  //learn NB

    infile.open("testSet.txt");
    std::vector<std::string> test_sentences = extract_sentences(infile);
    std::vector<std::vector<std::string>> test_data = itemize(test_sentences);  //format testing input
    std::vector<std::vector<int>> pre_processed_test = generate_feature_vector(test_data, vocab);  //generate feature vector from testing data


    double percent_correct_train = prediction_percent_correct(p_pos, p_neg, pre_processed_train, word_counts);
    double percent_correct_test = prediction_percent_correct(p_pos, p_neg, pre_processed_test, word_counts);

    std::ofstream outfile;
    outfile.open("preprocessed_train.txt");
    output_featurized_vector(outfile, pre_processed_train, vocab);
    outfile.close();

    outfile.open("preprocessed_test.txt");
    output_featurized_vector(outfile, pre_processed_test, vocab);
    outfile.close();

    std::cout << "Training accuracy: " << percent_correct_train * 100 << "%\n";
    std::cout << "Testing accuracy " << percent_correct_test * 100 << "%\n";
}
