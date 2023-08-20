#include "pch.h"
#include "../DBSCAN_new/DBSCAN.hpp"
#include "../DBSCAN_new/Operations.hpp"

#include <Eigen/Dense>
#include <random>
#include <Eigen/Core>
#include <unordered_map>


TEST(operations, normalization)
{
	Eigen::Matrix<double, 10, 4> data;
	data << 0.0, 0.0, 1.0, 1.0,
		0.0, 0.0, 1.0, 0.9,
		0.0, 0.0, 0.9f, 0.9,
		0.0, 0.0, 0.0, 0.9,
		0.0, 0.0, 0.0, 0.8,
		0.0, 0.0, 0.1, 0.8,
		0.0, 0.0, 0.1, 0.8,
		0.25, 0.0, 0.1, 0.8,
		0.23, 0.0, 0.1, 0.8,
		0.4, 0.0, 0.1, 0.8;

	std::vector<double> mins = { -1, 0, -2, 0 };
	std::vector<double> maxs = { 0, 2, 5, 3.5 };

	EXPECT_TRUE( Algorithm::Operations::denormalize(data, mins, maxs) );
	for (int i = 0; i < data.rows(); i++)
	{
		for (int j = 0; j < data.cols(); j++)
		{
			std::cout << "Max: " << maxs[j] << " data: " << data(i, j) << " Min: " << mins[j] << std::endl;
			std::cout << i << " " << j << std::endl;
			EXPECT_TRUE( maxs[j] >= data(i, j) >= mins[j] );
		}
	}
}

TEST(unsupervised, DBSCAN_basic_euclidean) 
{
	Eigen::Matrix<double, 10, 4> data;
	data << 0.0,  0.0,  1.0,  1.0,
			0.0,  0.0,  1.0,  0.9,
			0.0,  0.0,  0.9f, 0.9,
			0.0,  0.0,  0.0,  0.9,
			0.0,  0.0,  0.0,  0.8,
			0.0,  0.0,  0.1,  0.8,
			0.0,  0.0,  0.1,  0.8,
			0.25, 0.0,  0.1,  0.8,
			0.23, 0.0,  0.1,  0.8,
			0.4,  0.0,  0.1,  0.8;

	Algorithm::DBSCAN<Eigen::MatrixXd> clustering_algorithm(data,
			std::make_unique<Algorithm::Distance::Euclidean_Distance>());
	
	Eigen::VectorXi expected_output(10);
	expected_output << 1, 1, 1, 2, 2, 2, 2, 3, 3, 3;

	Eigen::VectorXi output;
	EXPECT_TRUE(clustering_algorithm.cluster(output, 0.2, 2));

	EXPECT_TRUE(output.size() == expected_output.size());
	for (int i = 0; i < expected_output.size(); i++)
	{
		EXPECT_TRUE(output(i) == expected_output(i));
	}
}

TEST(unsupervised, DBSCAN_basic_noise_point)
{
	Eigen::Matrix<double, 10, 4> data;
	data << 0.0, 0.0, 1.0, 1.0,
		0.0, 0.0, 1.0, 0.9,
		0.0, 0.0, 0.9f, 0.9,
		0.0, 0.3, 0.3, 0.9,
		0.0, 0.0, 0.0, 0.8,
		0.0, 0.0, 0.1, 0.8,
		0.0, 0.0, 0.1, 0.8,
		0.25, 0.0, 0.1, 0.8,
		0.23, 0.0, 0.1, 0.8,
		0.4, 0.0, 0.1, 0.8;

	Algorithm::DBSCAN<Eigen::MatrixXd> clustering_algorithm(data,
		std::make_unique<Algorithm::Distance::Euclidean_Distance>());

	Eigen::VectorXi expected_output(10);
	expected_output << 1, 1, 1, 0, 2, 2, 2, 3, 3, 3;

	Eigen::VectorXi output;
	EXPECT_TRUE(clustering_algorithm.cluster(output, 0.2, 2));

	EXPECT_TRUE(output.size() == expected_output.size());
	for (int i = 0; i < expected_output.size(); i++)
	{
		EXPECT_TRUE(output(i) == expected_output(i));
	}
}

TEST(unsupervised, DBSCAN_basic_chebychev)
{
	Eigen::Matrix<double, 10, 4> data;
	data << 0.0, 0.0, 1.0, 1.0,
		0.0, 0.0, 1.0, 0.9,
		0.0, 0.0, 0.9f, 0.9,
		0.0, 0.3, 0.3, 0.9,
		0.0, 0.0, 0.0, 0.8,
		0.0, 0.0, 0.1, 0.8,
		0.0, 0.0, 0.1, 0.8,
		0.25, 0.0, 0.1, 0.8,
		0.23, 0.0, 0.1, 0.8,
		0.4, 0.0, 0.1, 0.8;

	Algorithm::DBSCAN<Eigen::MatrixXd> clustering_algorithm(data,
		std::make_unique<Algorithm::Distance::Chebychev_Distance>());

	Eigen::VectorXi expected_output(10);
	expected_output << 1, 1, 1, 0, 2, 2, 2, 3, 3, 3;

	Eigen::VectorXi output;
	EXPECT_TRUE(clustering_algorithm.cluster(output, 0.2, 2));

	EXPECT_TRUE(output.size() == expected_output.size());
	for (int i = 0; i < expected_output.size(); i++)
	{
		EXPECT_TRUE(output(i) == expected_output(i));
	}
}

TEST(semisupervised, DBSCAN_basic_noise_point)
{
	Eigen::Matrix<double, 10, 4> data;
	data << 0.0, 0.0, 1.0, .95,
		0.0, 0.0, 1.0, 0.9,
		0.0, 0.0, 0.9f, 0.9,
		0.0, 0.3, 0.3, 0.9,
		0.0, 0.0, 0.0, 0.81,
		0.0, 0.0, 0.1, 0.8,
		0.0, 0.0, 0.1, 0.8,
		0.26, 0.0, 0.1, 0.79,
		0.23, 0.0, 0.1, 0.8,
		0.4, 0.0, 0.1, 0.8;

	Eigen::Matrix<double, 3, 4> supervision_data;
	supervision_data << 0.0, 0.0, 1.0, 1.0,
						0.0, 0.0, 0.0, 0.8,
						0.25, 0.0, 0.1, 0.8;
	Eigen::VectorXi supervision_labels(3);
	supervision_labels << 1, 2, 3;

	Algorithm::DBSCAN<Eigen::MatrixXd> clustering_algorithm(data,
		supervision_data,
		supervision_labels,
		std::make_unique<Algorithm::Distance::Euclidean_Distance>());

	Eigen::VectorXi expected_output(10);
	expected_output << 1, 1, 1, 0, 2, 2, 2, 3, 3, 3;

	Eigen::VectorXi output;
	EXPECT_TRUE(clustering_algorithm.cluster(output, 0.2, 2));

	EXPECT_TRUE(output.size() == expected_output.size());
	for (int i = 0; i < expected_output.size(); i++)
	{
		EXPECT_TRUE(output(i) == expected_output(i));
	}
}