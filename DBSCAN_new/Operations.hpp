#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include <Eigen/Dense>
#include <vector>


namespace Algorithm
{
	namespace Operations
	{
		template <typename T,
				  typename std::enable_if_t<std::is_base_of<Eigen::MatrixBase<T>, T>::value, int> = 0, 
			      typename U>
		void normalize(T& data, std::vector<U>& minimum_value, std::vector<U>& maximum_value)
		{
			minimum_value.clear();
			maximum_value.clear();

			// Initialize Maximums/Minimums
			for (int i = 0; i < data.cols(); i++)
			{
				maximum_value.push_back(std::numeric_limits<double>::min());
				minimum_value.push_back(std::numeric_limits<double>::max());
			}

			// Loop through entire data and modify maximums/minimums
			for (int i = 0; i < data.rows(); i++)
			{
				for (int j = 0; j < data.cols(); j++)
				{
					maximum_value[j] = std::max(maximum_value[j], data(i, j));
					minimum_value[j] = std::min(minimum_value[j], data(i, j));
				}
			}

			// Define all denominators so that they are not recomputed in nested for loops
			std::vector<U> denominators;
			for (int i = 0; i < data.cols(); i++)
			{
				denominators.push_back(maximum_value[i] - minimum_value[i]);
			}

			for (int i = 0; i < data.rows(); i++)
			{
				for (int j = 0; j < data.cols(); j++)
				{
					data(i, j) = (data(i, j) - minimum_value[j]) / denominators[j];
				}
			}
		}

		template <typename T,
			typename std::enable_if_t<std::is_base_of<Eigen::MatrixBase<T>, T>::value, int> = 0,
			typename U>
			void normalize(T& data, 
						   T& supervision_data,
						   std::vector<U>& minimum_value, 
						   std::vector<U>& maximum_value)
		{
			minimum_value.clear();
			maximum_value.clear();

			// Initialize Maximums/Minimums
			for (int i = 0; i < data.cols(); i++)
			{
				maximum_value.push_back(std::numeric_limits<double>::min());
				minimum_value.push_back(std::numeric_limits<double>::max());
			}

			// Loop through entire data and modify maximums/minimums
			for (int i = 0; i < data.rows(); i++)
			{
				for (int j = 0; j < data.cols(); j++)
				{
					maximum_value[j] = std::max(maximum_value[j], data(i, j));
					minimum_value[j] = std::min(minimum_value[j], data(i, j));
				}
			}

			// Loop through entire supervision data set and modify max/mins
			for (int i = 0; i < supervision_data.rows(); i++)
			{
				for (int j = 0; j < supervision_data.cols(); j++)
				{
					maximum_value[j] = std::max(maximum_value[j], supervision_data(i, j));
					minimum_value[j] = std::min(minimum_value[j], supervision_data(i, j));
				}
			}

			// Define all denominators so that they are not recomputed in nested for loops
			std::vector<U> denominators;
			for (int i = 0; i < data.cols(); i++)
			{
				denominators.push_back(maximum_value[i] - minimum_value[i]);
			}

			for (int i = 0; i < data.rows(); i++)
			{
				for (int j = 0; j < data.cols(); j++)
				{
					data(i, j) = (data(i, j) - minimum_value[j]) / denominators[j];
				}
			}

			// Now normalizing supervision data set
			for (int i = 0; i < supervision_data.rows(); i++)
			{
				for (int j = 0; j < supervision_data.cols(); j++)
				{
					supervision_data(i, j) = (supervision_data(i, j) - minimum_value[j]) / denominators[j];
				}
			}
		}

		template <typename T,
				  typename std::enable_if_t<std::is_base_of<Eigen::MatrixBase<T>, T>::value, int> = 0,
				  typename U>
		bool denormalize(T& data, const std::vector<U>& minimum_value, const std::vector<U>& maximum_value)
		{
			bool success = false;

			if (data.cols() == minimum_value.size() && data.cols() == maximum_value.size())
			{
				success = true;
				for (int i = 0; i < data.rows(); i++)
				{
					for (int j = 0; j < data.cols(); j++)
					{
						data(i, j) = (data(i, j) * (maximum_value[j] - minimum_value[j]) + minimum_value[j]);
					}
				}
			}

			return success;
		}
	}
}

#endif