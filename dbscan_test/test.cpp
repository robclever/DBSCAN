#include "pch.h"
#include "../DBSCAN_new/DBSCAN.hpp"

#include <Eigen/Dense>
#include <random>
#include <Eigen/Core>
#include <unordered_map>

TEST(clustering, DBSCAN_basic_euclidean) 
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

	Eigen::VectorXi output = clustering_algorithm.perform_batch_cluster(0.2, 2);

	EXPECT_TRUE(output.size() == expected_output.size());
	for (int i = 0; i < expected_output.size(); i++)
	{
		EXPECT_TRUE(output(i) == expected_output(i));
	}
}

TEST(clustering, DBSCAN_basic_noise_point)
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

	Eigen::VectorXi output = clustering_algorithm.perform_batch_cluster(0.2, 2);

	EXPECT_TRUE(output.size() == expected_output.size());
	for (int i = 0; i < expected_output.size(); i++)
	{
		EXPECT_TRUE(output(i) == expected_output(i));
	}
}

TEST(clustering, DBSCAN_basic_chebychev)
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

	Eigen::VectorXi output = clustering_algorithm.perform_batch_cluster(0.2, 2);

	EXPECT_TRUE(output.size() == expected_output.size());
	for (int i = 0; i < expected_output.size(); i++)
	{
		EXPECT_TRUE(output(i) == expected_output(i));
	}
}