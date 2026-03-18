/*
 * Custom ElementCamera that can be disabled during comparison slider drag
 */

#pragma once

#include <nvapp/elem_camera.hpp>
#include <functional>

namespace vk_gaussian_splatting {

class ElementCameraCustom : public nvapp::ElementCamera
{
public:
  ElementCameraCustom(std::shared_ptr<nvutils::CameraManipulator> camera = nullptr)
      : ElementCamera(camera)
  {
  }

  // Set a callback to check if camera updates should be disabled
  void setDisableCallback(std::function<bool()> callback) { m_disableCallback = callback; }

  void onUIRender() override
  {
    // Check if camera updates should be disabled
    if(m_disableCallback && m_disableCallback())
    {
      // Skip camera update when comparison slider is being dragged
      return;
    }

    // Otherwise, call the base class implementation
    nvapp::ElementCamera::onUIRender();
  }

private:
  std::function<bool()> m_disableCallback;
};

}  // namespace vk_gaussian_splatting
