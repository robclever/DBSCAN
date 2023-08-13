#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#include <Eigen/Dense>
#include <vector>
#include <limits>
#include "Distance_strategies.hpp"

namespace Algorithm
{
    constexpr int UNDEFINED_LABEL   = -1;
    constexpr int NOISE_LABEL       =  0;

    /**
     * DBSCAN algorithm interface that provides context for distance function to be selected
     * on construction
     */
    template <typename T,
        typename std::enable_if_t<std::is_base_of<Eigen::MatrixBase<T>, T>::value, int> = 0>
    class DBSCAN
    {
        /**
         * @var Strategy The Context maintains a reference to one of the Strategy
         * objects. The Context does not know the concrete class of a strategy. It
         * should work with all strategies via the Strategy interface.
         */
    private:
        // The strategy used to cluster the data
        std::unique_ptr<Algorithm::Distance::Distance_Function> distance_func_;

        // The data that is eventually clustered
        T input_data;

        void normalize_data()
        {
            std::vector<double> maximums;
            std::vector<double> minimums;
            // Initialize Maximums/Minimums
            for (int i = 0; i < input_data.cols(); i++)
            {
                maximums.push_back(std::numeric_limits<double>::min());
                minimums.push_back(std::numeric_limits<double>::max());
            }

            // TODO:
        }

    public:
        explicit DBSCAN(T input,
                        std::unique_ptr<Algorithm::Distance::Distance_Function>&& func = {}) :
            distance_func_(std::move(func)),
            input_data(input)
        {   
            // If the input data is not on scale [0, 1], normalization will need to be performed
        }

        /**
         * Usually, the Context allows replacing a Strategy object at runtime.
         */
        void set_distance_function(std::unique_ptr<Algorithm::Distance::Distance_Function>&& distance_function)
        {
            distance_func_ = std::move(distance_function);
        }

        /**
         * The Context delegates some work to the Strategy object instead of
         * implementing +multiple versions of the algorithm on its own.
         */
        Eigen::VectorXi perform_batch_cluster(const double& epsilon, const int& min_points) const
        {
            Eigen::VectorXi label_data = Eigen::VectorXi::Constant(input_data.rows(), UNDEFINED_LABEL);
            int counter_cluster = 0;
            std::vector<int> neighbors_vect;

            for (int i = 0; i < input_data.rows(); i++)
            {
                // If the label is not set for the current point, attempt to cluster it
                if (label_data(i) == UNDEFINED_LABEL)
                {
                    this->range_linear_scan(i, 
                                            epsilon,
                                            neighbors_vect);

                    if (neighbors_vect.size() < min_points)
                    {
                        label_data(i) = NOISE_LABEL;
                    }
                    else
                    {
                        counter_cluster += 1;

                        // Find neighboring points that belong in the cluster
                        this->expand_cluster(label_data,
                                             input_data,
                                             neighbors_vect,
                                             counter_cluster,
                                             epsilon,
                                             min_points);
                    }
                }
                neighbors_vect.clear();
            }
            
            return label_data;
        }

        Eigen::VectorXi perform_sequential_cluster(const double& epsilon, 
                                                   const int& min_points,
                                                   const Eigen::VectorXi data_labels,
                                                   const T& additional_points) const
        {
            // TODO
            return Eigen::VectorXi::Zero(1);
        }

        void range_linear_scan(const int& this_point,
                               const double& epsilon,
                               std::vector<int>& neighbors) const
        {
            for (int point_indx = 0; point_indx < input_data.rows(); point_indx++)
            {
                if (distance_func_->compute(input_data.row(this_point), 
                                            input_data.row(point_indx)) <= epsilon)
                {
                    neighbors.push_back(point_indx);
                }
            }
        }

        void expand_cluster(Eigen::VectorXi& label_data,
                            const Eigen::MatrixXd& input_data,
                            const std::vector<int>& neighbors,
                            const int& next_cluster_label,
                            const double& epsilon,
                            const int& min_pts) const
        {
            for (int point_indx = 0; point_indx < neighbors.size(); point_indx++)
            {
                if (label_data(neighbors[point_indx]) == UNDEFINED_LABEL)
                {
                    // Find other neighboring points that meet the distance criteria for this point
                    std::vector<int> this_point_neighbors;
                    range_linear_scan(neighbors[point_indx], 
                                      epsilon, 
                                      this_point_neighbors);

                    if (this_point_neighbors.size() >= min_pts)
                    {
                        if (label_data(neighbors[point_indx]) == UNDEFINED_LABEL)
                        {
                            label_data(neighbors[point_indx]) = next_cluster_label;
                        }
                    }
                }
            }
        }
    };


};

#endif //