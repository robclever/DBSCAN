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

    using d_func      = Algorithm::Distance::Distance_Function;
    using eig_vectori = Eigen::VectorXi;

    template <typename T,
        typename std::enable_if_t<std::is_base_of<Eigen::MatrixBase<T>, T>::value, int> = 0>
    class Cluster_Result
    {
        bool                    data_normalized;
        T                       input_data;
        eig_vectori             output_labels;
        std::unique_ptr<d_func> function;
        int                     unique_clusters;
        double                  epsilon;
        int                     minimum_points;
    };

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

        bool data_normalized;

        std::vector<double> maximums;
        std::vector<double> minimums;

    public:
        explicit DBSCAN(T input,
                        std::unique_ptr<d_func>&& func = {}) :
            distance_func_(std::move(func)),
            input_data(input),
            data_normalized(false)
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

        bool is_data_scaled() const
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
         * TODO: Allow user to modify data
         */


        /**
         * Implementation for DBSCAN "batch" clustering. This algorithm assumes that all data has been collected
         * and sent as a whole to the implementation, given on construction.
         *   
         */
        eig_vectori cluster_batch(const double& epsilon, const int& min_points) const
        {
            eig_vectori label_data = eig_vectori::Constant(input_data.rows(), UNDEFINED_LABEL);
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

        eig_vectori cluster_batch(const T& supervised_data, const eig_vectori& input_labels, const double& epsilon, const int& min_points) const
        {
            eig_vectori label_data = eig_vectori::Constant(input_data.rows(), UNDEFINED_LABEL);
            // Start after the labels for new clusters
            int counter_cluster = std::max(input_labels) + 1;
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
                        // Try to fit data first to supervised dataset
                        int matching_index = -1;
                        eig_vectori matching_cluster = eige_vectori::Zeros(3);
                        if (supervised_data_scan(supervised_data, i, epsilon, matching_index))
                        {
                            matching_cluster = input_labels(matching_index);
                        }
                        else
                        {
                            counter_cluster += 1;
                            matching_cluster = counter_cluster;
                        }

                        // Find neighboring points that belong in the cluster
                        this->expand_cluster(label_data,
                            input_data,
                            neighbors_vect,
                            matching_cluster,
                            epsilon,
                            min_points);
                    }
                }
                neighbors_vect.clear();
            }

            return label_data;
        }

        eig_vectori cluster_sequential(const double& epsilon, 
                                       const int& min_points,
                                       const eig_vectori data_labels,
                                       const T& additional_points) const
        {
            // TODO
            return eig_vectori::Zero(1);
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

        void expand_cluster(eig_vectori& label_data,
                            const T& input_data,
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