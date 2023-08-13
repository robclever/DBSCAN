#ifndef DISTANCE_STRATEGIES_HPP
#define DISTANCE_STRATEGIES_HPP

#include <Eigen/Dense>

namespace Algorithm
{
    namespace Distance
    {
        class Distance_Function
        {
        public:
            virtual ~Distance_Function() = default;

            virtual double compute(const Eigen::VectorXd& this_point, const Eigen::VectorXd& neighbor_point) const = 0;
        };

        /**
         * Concrete Strategies implement the algorithm while following the base Strategy
         * interface. The interface makes them interchangeable in the Context.
         */
        class Euclidean_Distance : public Distance_Function
        {
        public:
            double compute(const Eigen::VectorXd& this_point, const Eigen::VectorXd& neighbor_point) const override
            {
                double output = 0.0;
                for (int i = 0; i < this_point.size(); i++)
                {
                    output += std::pow((this_point(i) - neighbor_point(i)), 2);
                }

                return sqrt(output);
            }
        };

        class Chebychev_Distance : public Distance_Function
        {
            double compute(const Eigen::VectorXd& this_point, const Eigen::VectorXd& neighbor_point) const override
            {
                double output = 0.0;
                for (int i = 0; i < this_point.size(); i++)
                {
                    output = std::max( output, std::abs(this_point(i) - neighbor_point(i)) );
                }

                return output;
            }
        };

        class Bhattacharyya_Distance : public Distance_Function
        {
            double compute(const Eigen::VectorXd& this_point, const Eigen::VectorXd& neighbor_point) const override
            {
                return 0.0;
            }
        };
    };
}

#endif