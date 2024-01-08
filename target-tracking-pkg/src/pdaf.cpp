#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <vortex_filtering/vortex_filtering.hpp>

using std::string;
// using std::endl;

class PDAF {
public:
    using DynModT = vortex::models::ConstantVelocity<2>;
    using SensModT = vortex::models::IdentitySensorModel<4, 2>;
    using DynModPtr = std::shared_ptr<DynModT>;
    using SensModPtr = std::shared_ptr<SensModT>;
    using EKF = vortex::filter::EKF_M<DynModT, SensModT>;
    using Gauss2d = vortex::prob::MultiVarGauss2d;
    using Gauss4d = vortex::prob::MultiVarGauss4d;
    using Measurements2d = std::vector<Eigen::Vector2d>;

    DynModPtr dynamic_model_;
    SensModPtr sensor_model_;
    double gate_threshold_;

    PDAF(DynModPtr dynamic_model, SensModPtr sensor_model)
        : dynamic_model_(dynamic_model)
        , sensor_model_(sensor_model)
        , gate_threshold_(2)
    {
        std::cout << "Created PDAF class with given models!" << std::endl;
    }

    PDAF() = default;

    // predict next state if dynamic model and sensor model are defined
    void predict_next_state(Gauss4d x_est, std::vector<Eigen::Vector2d> z_meas, double timestep)
    {
        if (!dynamic_model_ || !sensor_model_) {
            throw std::runtime_error("Dynamic model or sensor model not set");
        }
        predict_next_state(x_est, z_meas, timestep, dynamic_model_, sensor_model_);
    }

    // predict next state, if ekf is not defined
    void predict_next_state(Gauss4d x_est, Measurements2d z_meas, double timestep, DynModPtr dyn_model, SensModPtr sen_model)
    {
        EKF ekf;
        //std::pair<Gauss4d, Gauss2d> x_z_pred = ekf.predict(dyn_model, sen_model, timestep, x_est);
        auto [x_pred, z_pred] = ekf.predict(dyn_model, sen_model, timestep, x_est);
        Measurements2d filtered = apply_gate(gate_threshold_, z_meas, z_pred);

        // x_post -> vector
        for (const auto& measurement : filtered) {
            // x_post append with:
            ekf.update(x_pred, z_pred, measurement);
        }
    }

    Measurements2d apply_gate(double gate_threshold, Measurements2d z_meas, Gauss2d z_pred)
    {
        Measurements2d filtered_meas;

        for (const auto& measurement : z_meas) {
            double distance = z_pred.mahalanobis_distance(measurement);

            if (distance < gate_threshold) {
                filtered_meas.push_back(measurement);
            }
        }

        return filtered_meas;
    }
};

int main(int argc, char** argv)
{
    PDAF pdaf_class();
    return 0;
}