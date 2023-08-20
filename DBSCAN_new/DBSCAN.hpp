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

    using d_func    = Algorithm::Distance::Distance_Function;
    using EVector_i = Eigen::VectorXi;

    /**
     * DBSCAN algorithm interface that provides context for distance function to be selected
     * on construction
     * 
     * TODO: Make separate derived object for semisupervised DBSCAN implementation that allows 
     * user to specify some initial data and labels that correspond with the data.
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
        std::unique_ptr<d_func> distance_func_;

        // The data that is eventually clustered
        T input_data;

        // Feature data that has labels corresponding to it
        T supervision_data;
        
        // Labels for the supervision data member
        EVector_i labels;

        // If true, the input data was normalized. If false, data was already normalized pre-construction
        bool data_normalized;

        bool has_labels;

        std::vector<double> maximums;
        std::vector<double> minimums;

        // Tries to find a label that most closely matches the input data set to the supervision data set.
        // If the point does not satisfy and supervision label, then the function will return false.
        bool supervision_scan(const int& this_point,
                              const double& epsilon,
                              int& best_matching_point) const
        {
            double lowest_eps = std::numeric_limits<double>::max();
            bool success      = false;
            for (int supervision_indx = 0; supervision_indx < supervision_data.rows(); supervision_indx++)
            {
                double distance = distance_func_->compute(input_data.row(this_point),
                    supervision_data.row(supervision_indx));
                if (distance <= epsilon && distance < lowest_eps)
                {
                    lowest_eps = distance;
                    best_matching_point = supervision_indx;
                    success = true;
                }
            }

            if (!success)
            {
                best_matching_point = -1;
            }

            return success;
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

        // Function used to find and label all neighboring points that meet DBSCAN distance criteria.
        void expand_cluster(EVector_i& label_output,
                            const T& input_data,
                            const std::vector<int>& neighbors,
                            const int& next_cluster_label,
                            const double& epsilon,
                            const int& min_pts) const
        {
            for (int point_indx = 0; point_indx < neighbors.size(); point_indx++)
            {
                if (label_output(neighbors[point_indx]) == UNDEFINED_LABEL)
                {
                    // Find other neighboring points that meet the distance criteria for this point
                    std::vector<int> this_point_neighbors;
                    range_linear_scan(neighbors[point_indx],
                                      epsilon,
                                      this_point_neighbors);

                    if (this_point_neighbors.size() >= min_pts)
                    {
                        if (label_output(neighbors[point_indx]) == UNDEFINED_LABEL)
                        {
                            label_output(neighbors[point_indx]) = next_cluster_label;
                        }
                    }
                }
            }
        }

    public:
        DBSCAN(T input,
               std::unique_ptr<d_func>&& func = {}) :
            distance_func_(std::move(func)),
            input_data(input),
            data_normalized(false),
            has_labels(false)
        {   
            // If the input data is not on scale [0, 1], normalization will need to be performed
            for (int i = 0; i < input_data.rows() && data_normalized == false; i++)
            {
                for (int j = 0; j < input_data.cols() && data_normalized == false; j++)
                {
                    // Data should be in the range of [0, 1] if already normalized
                    if ( !(input_data(i, j) >= 0 && input_data(i, j) <= 1) )
                    {
                        // This class needs to normalize the data prior to use
                        data_normalized = true;
                    }
                }
            }

            // Scale data to range [0, 1]
            if (data_normalized)
            {
                Algorithm::Operations::normalize(input_data, minimums, maximums);
            }
        }

        DBSCAN(T input,
               T supervision_data,
               EVector_i supervision_labels,
               std::unique_ptr<d_func>&& func = {}):
            distance_func_(std::move(func)),
            input_data(input),
            supervision_data(supervision_data),
            labels(supervision_labels),
            data_normalized(false),
            has_labels(true)
        {
            // If the input data is not on scale [0, 1], normalization will need to be performed
            for (int i = 0; i < input_data.rows() && data_normalized == false; i++)
            {
                for (int j = 0; j < input_data.cols() && data_normalized == false; j++)
                {
                    // Data should be in the range of [0, 1] if already normalized
                    if (!(input_data(i, j) >= 0 && input_data(i, j) <= 1))
                    {
                        // This class needs to normalize the data prior to use
                        data_normalized = true;
                    }
                }
            }

            // Scale data to range [0, 1]
            if (data_normalized)
            {
                Algorithm::Operations::normalize(input_data, supervision_data, minimums, maximums);
            }
        }

        bool is_data_normalized() const
        {
            return this->data_normalized;
        }

        /**
         * DBSCAN allows replacing a Strategy object at runtime.
         */
        void set_distance_function(std::unique_ptr<d_func>&& distance_function)
        {
            distance_func_ = std::move(distance_function);
        }

        /**
         * Implementation for DBSCAN "batch" clustering. This algorithm assumes that all data has been collected
         * and sent as a whole to the implementation, given on construction.
         *   
         * This specific overload performs clustering over the entire data set. It can perform unsupervised or semi-
         * supervised learning based on the constructor used to assign labels/training data.
         */
        EVector_i cluster(const double& epsilon, const int& min_points) const
        {
            EVector_i label_output = EVector_i::Constant(input_data.rows(), UNDEFINED_LABEL);

            int counter_cluster;

            if (has_labels)
            {

                // Start after the labels for new clusters for SS
                counter_cluster = labels.maxCoeff() + 1;
            }
            else
            {
                // Start cluster labels at 0 for Unsupervised
                counter_cluster = 0;
            }

            std::vector<int> neighbors_vect;

            for (int i = 0; i < input_data.rows(); i++)
            {
                // If the label is not set for the current point, attempt to cluster it
                if (label_output(i) == UNDEFINED_LABEL)
                {
                    this->range_linear_scan(i,
                                            epsilon,
                                            neighbors_vect);

                    if (neighbors_vect.size() < min_points)
                    {
                        label_output(i) = NOISE_LABEL;
                    }
                    else
                    {
                        // Semi-supervised learning
                        if (has_labels)
                        {
                            // Try to fit data first to supervised dataset
                            int matching_index;
                            int matching_cluster;
                            if (supervision_scan(i, epsilon, matching_index))
                            {
                                counter_cluster = labels(matching_index);
                            }
                            else
                            {
                                counter_cluster += 1;
                            }
                        }
                        // Unsupervised Learning
                        else
                        {
                            counter_cluster += 1;
                        }

                        // Find neighboring points that belong in the cluster
                        this->expand_cluster(label_output,
                                             input_data,
                                             neighbors_vect,
                                             counter_cluster,
                                             epsilon,
                                             min_points);
                    }
                }

                neighbors_vect.clear();
            }

            return label_output;
        }
    };

};

#endif //