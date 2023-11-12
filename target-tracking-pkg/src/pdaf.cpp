#include <cstdio>
#include <Eigen/Dense>
#include <string>
#include <iostream>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

using std::string;
//using std::endl;

class PDAF {
  public:
  float time_step;
  Eigen::Matrix4d apply_velocity;
  Eigen::Matrix<double, 2, 4> C;
  vortex::prob::MultiVarGauss<4> posterior_state_estimate;
  vortex::prob::MultiVarGauss<4> prior_state_estimate;
  vortex::prob::MultiVarGauss<2> predicted_observation;
  vortex::prob::MultiVarGauss<4> model_disturbance;
  vortex::prob::MultiVarGauss<2> measurment_noise;
  int validation_scaling_param;
  float minimal_mahalanobis_distance;
  float p_no_match;

  PDAF(float time_step, Eigen::Vector4d state_post, Eigen::Matrix4d P_post, Eigen::Matrix4d Q, Eigen::Matrix2d R,
  int val_scal_param, float min_mahal_dist, float p_n_match) 
    : posterior_state_estimate(state_post, P_post), 
    prior_state_estimate(state_post, P_post),
    predicted_observation(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()),
    model_disturbance(Eigen::Vector4d::Zero(), Q),
    measurment_noise(Eigen::Vector2d::Zero(), R)
  {

    time_step = time_step;
    validation_scaling_param = val_scal_param;
    minimal_mahalanobis_distance = min_mahal_dist;
    p_no_match = p_n_match;

    C << 1.0, 0, 0, 0,
         0, 1.0, 0, 0;

    apply_velocity << 1.0, 0, time_step, 0, 
                    0, 1.0, 0, time_step,
                    0, 0, 1.0, 0,
                    0, 0, 0, 1.0;

    std::cout << predicted_observation.mean() << std::endl;
  }

  void prediction_step()
  {
    vortex::prob::MultiVarGauss<4> prior_state_estimate_new(
      apply_velocity * posterior_state_estimate.mean(), 
      apply_velocity * posterior_state_estimate.cov() * apply_velocity.transpose() + model_disturbance.cov()
    );
    prior_state_estimate = prior_state_estimate_new;

    vortex::prob::MultiVarGauss<2> predicted_observation_new(
      C * prior_state_estimate.mean(), 
      C * prior_state_estimate.cov() * C.transpose() + measurment_noise.cov()
    );
    predicted_observation = predicted_observation_new;
  }

  void update_model(float time_step_new) {
    time_step = time_step_new;
    apply_velocity << 1.0, 0, time_step, 0, 
                    0, 1.0, 0, time_step,
                    0, 0, 1.0, 0,
                    0, 0, 0, 1.0;
    std::cout << "Updating model with time_step="<< time_step << "\n" << std::endl;
  }

  Eigen::Matrix4d compute_kalman_gain()
  {
    // Eigen::Matrix4d C_P_CT = C * prior_state_estimate.cov() * C.transpose();
    // Eigen::Matrix4d C_P_CT_cov = C_P_CT + measurment_noise.cov();
    // Eigen::Matrix4d L = prior_state_estimate.cov() * C.transpose() * C_P_CT_cov.inverse();

    return Eigen::Matrix4d::Identity();
  }
};

int main(int argc, char ** argv) 
{

  // predefined configuration
  // for more information lock at :
  // https://github.com/vortexntnu/vortex-cv/blob/main/tracking/config/config_traking_sys.yaml
  float time_step = 0.1f;
  int validation_gate_scaling_param = 2;
  float minimal_mahalanobis_distance = 0.001f;
  float p_no_match = 0.5f;
  Eigen::Vector4d state_post = Eigen::Vector4d::Zero();
  Eigen::Matrix4d P_post = Eigen::Matrix4d::Zero();
  Eigen::Matrix4d Q = Eigen::Matrix4d::Identity();
  Eigen::Matrix2d R = Eigen::Matrix2d::Zero();


  PDAF pdaf_class(time_step, state_post, P_post, Q, R, validation_gate_scaling_param, minimal_mahalanobis_distance, p_no_match);

  
  std::cout << pdaf_class.apply_velocity << "\n" << std::endl;
  std::cout << pdaf_class.C << "\n" << std::endl;

  pdaf_class.update_model(5.0f);

  std::cout << pdaf_class.apply_velocity << "\n" << pdaf_class.time_step << "\n" << std::endl;

  (void) argc;
  (void) argv;

  printf("hello world target-tracking-pkg package\n");



  return 0;
}
