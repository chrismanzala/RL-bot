// Minimal ROS 2 (rclcpp) node: drives turtlesim's turtle1 in a circle.
// Replaces the old ROS 1 stub. Later this node will run a trained swarm policy.
//
// To build, copy it into an ament_cmake package that depends on rclcpp and
// geometry_msgs (ros2 pkg create --build-type ament_cmake --node-name swarm_bridge
// swarm_bridge --dependencies rclcpp geometry_msgs), build with colcon, then:
//   ros2 run turtlesim turtlesim_node
//   ros2 run swarm_bridge swarm_bridge
#include <chrono>
#include <memory>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class SwarmBridge : public rclcpp::Node {
 public:
  SwarmBridge() : Node("swarm_bridge") {
    pub_ = create_publisher<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10);
    timer_ = create_wall_timer(100ms, [this] {
      geometry_msgs::msg::Twist cmd;
      cmd.linear.x = 1.0;   // m/s forward
      cmd.angular.z = 0.5;  // rad/s turn
      pub_->publish(cmd);
    });
    RCLCPP_INFO(get_logger(), "Publishing to /turtle1/cmd_vel at 10 Hz");
  }

 private:
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SwarmBridge>());
  rclcpp::shutdown();
  return 0;
}
