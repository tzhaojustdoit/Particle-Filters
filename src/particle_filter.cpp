/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang, Tianhua Zhao
 */

#include "particle_filter.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

static std::default_random_engine dre;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to
   * first position (based on estimates of x, y, theta and their uncertainties
   * from GPS) and all weights to 1.
   * Add random Gaussian noise to each particle.
   */

  if (is_initialized) {
    return;
  }

  num_particles = 100;  // set the number of particles
  particles.reserve(num_particles);

  // Create normal distribution for x, y, and theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    particles.emplace_back(i, dist_x(dre), dist_y(dre), dist_theta(dre), 1.0);
  }
  weights.resize(num_particles, 1);

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  // avoid division by 0
  if (fabs(yaw_rate) < 0.0001) {
    double v_by_t = velocity * delta_t;
    for (auto& p : particles) {
      p.x += v_by_t * cos(p.theta) + dist_x(dre);
      p.y += v_by_t * sin(p.theta) + dist_y(dre);
      p.theta += dist_theta(dre);
    }
  } else {
    double v_over_yaw_rate = velocity / yaw_rate;
    double yaw_rate_by_delta_t = yaw_rate * delta_t;
    for (auto& p : particles) {
      p.x += v_over_yaw_rate * (sin(p.theta + yaw_rate_by_delta_t) - sin(p.theta)) + dist_x(dre);
      p.y += v_over_yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate_by_delta_t)) + dist_y(dre);
      p.theta += yaw_rate_by_delta_t + dist_theta(dre);
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian
   * distribution. You can read more about this distribution here:
   * https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * The observations are given in the VEHICLE'S coordinate system.
   */
  int num_observation = observations.size();
  for (int j = 0; j < num_particles; j++) {
    auto& p = particles[j];
    // Get landmarks within sensor range for this particle
    vector<Map::single_landmark_s> landmarks_in_range;
    for (auto& landmark : map_landmarks.landmark_list) {
      if (dist(landmark.x_f, landmark.y_f, p.x, p.y) <= sensor_range) {
        landmarks_in_range.push_back(landmark);
      }
    }

    vector<double> sense_x(num_observation, 0);     // x locations for every observation in map coordiate system
    vector<double> sense_y(num_observation, 0);     // y locations for every observation in map coordiate system
    vector<int> associations(num_observation, -1);  // association for every observation

    double sin_theta = sin(p.theta);
    double cos_theta = cos(p.theta);
    for (int i = 0; i < num_observation; ++i) {
      auto& observation = observations[i];
      // transform the observation from the vehicle's coordinate system to the map's coordinate system.
      double x_map = observation.x * cos_theta - observation.y * sin_theta + p.x;
      double y_map = observation.x * sin_theta + observation.y * cos_theta + p.y;
      sense_x[i] = x_map;
      sense_y[i] = y_map;

      // Find the nearest landmark in range
      double min_dist = sensor_range;
      int association = -1;
      for (auto& landmark : landmarks_in_range) {
        double distance = dist(x_map, y_map, landmark.x_f, landmark.y_f);
        if (distance < min_dist) {
          association = landmark.id_i;
          min_dist = distance;
        }
      }

      // Set association
      associations[i] = association;
    }

    SetAssociations(p, associations, sense_x, sense_y);  // optional, for visualization in simulation

    // Compute Weights
    p.weight = 1;
    for (int i = 0; i < num_observation; ++i) {
      if (associations[i] == -1) {
        p.weight = 0;
        break;
      }
      auto& landmark = map_landmarks.landmark_list[associations[i] - 1];  // assuming landmark_list is ordered by id,
                                                                          // subtract 1 to get the index of an id
      p.weight *= multiv_prob(std_landmark[0], std_landmark[1], sense_x[i], sense_y[i], landmark.x_f, landmark.y_f);
    }
    // weights[j] = p.weight; // optional
  }
}

void ParticleFilter::resample() {
  vector<Particle> samples;
  samples.reserve(num_particles);
  std::discrete_distribution<int> dd(weights.begin(), weights.end());  // discrete distribution for resampling
  for (int i = 0; i < num_particles; ++i) {
    int idx = dd(dre);
    samples.push_back(particles[idx]);
  }
  particles = samples;
}

void ParticleFilter::SetAssociations(Particle& particle, const vector<int>& associations, const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
