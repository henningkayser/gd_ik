#include <gd_ik/fk_moveit.hpp>
#include <gd_ik/frame.hpp>

#include <algorithm>
#include <memory>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <vector>

namespace gd_ik {

auto make_fk_fn(std::shared_ptr<moveit::core::RobotModel const> robot_model,
                moveit::core::JointModelGroup const* jmg,
                std::vector<size_t> tip_link_indexes) -> FkFn {
    auto robot_state = moveit::core::RobotState(robot_model);
    robot_state.setToDefaultValues();

    auto const tip_link = robot_model->getLinkModel(tip_link_indexes.at(0));
    auto const tip_joint = tip_link->getParentJointModel();
    auto const joints = jmg->getActiveJointModels();

    if (jmg->isChain() &&  // All joints in line
        robot_model->getCommonRoot(joints.back(), tip_joint) == joints.back() &&  // Tip is child
        tip_link_indexes.size() == 1)                                             // Only one tip
    {
        // Compute the static parts of the FK solution (root->jmg, jmg->tip) (no need to have active
        // positions for that)
        // TODO: This needs the seed state in case there are actuated joints further up in the
        // chain, outside of jmg
        const Eigen::Isometry3d root_to_jmg_transform =
            robot_state.getGlobalLinkTransform(joints.front()->getParentLinkModel());
        // TODO: compute for each tip
        const Eigen::Isometry3d jmg_to_tip_transform =
            robot_state.getGlobalLinkTransform(joints.back()->getChildLinkModel()).inverse() *
            robot_state.getGlobalLinkTransform(tip_link);

        return [=](std::vector<double> const& active_positions) {
            // Iterate joints (this is important, since links may have multiple descendants)
            // NOTE: This implementation is based on the assumption that all joints are active
            // without fixed or mimic joints in between. If we want to support non-active joints,
            // we could extend active_positions with corresponding values in advance
            Eigen::Isometry3d fk_result = root_to_jmg_transform;
            Eigen::Isometry3d joint_transform;
            size_t joint_idx = 0;
            for (auto joint : joints) {
                joint->computeTransform(&active_positions[joint_idx++], joint_transform);
                fk_result = fk_result * joint->getChildLinkModel()->getJointOriginTransform() *
                            joint_transform;
            }

            fk_result = fk_result * jmg_to_tip_transform;

            const std::vector<Frame> tip_frames{Frame::from(fk_result)};
            return tip_frames;
        };
    } else {
        // IK function is mutable so it re-uses the robot_state instead of creating
        // new copies. This function should not be shared between threads.
        // It is however safe to make copies of.
        return [=](std::vector<double> const& active_positions) mutable {
            robot_state.setJointGroupPositions(jmg, active_positions);
            robot_state.updateLinkTransforms();

            std::vector<Frame> tip_frames;
            std::transform(tip_link_indexes.cbegin(),
                           tip_link_indexes.cend(),
                           std::back_inserter(tip_frames),
                           [&](auto index) {
                               auto const* link_model = robot_model->getLinkModel(index);
                               return Frame::from(robot_state.getGlobalLinkTransform(link_model));
                           });
            return tip_frames;
        };
    }
}

}  // namespace gd_ik
