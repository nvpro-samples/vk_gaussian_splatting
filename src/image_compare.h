#pragma once

#include <vulkan/vulkan.h>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/descriptors.hpp>
#include <nvslang/slang.hpp>
#include <string>

#include "image_compare_shaderio.h"

// Forward declarations for ImGui types
struct ImVec2;
struct ImDrawList;

namespace vk_gaussian_splatting {

// Forward declaration
class ImageCompareUI;

class ImageCompare
{
  // Allow ImageCompareUI to access private members for UI rendering
  friend class ImageCompareUI;

public:
  // ============================================================================
  // Nested Types (Enums and Structs)
  // ============================================================================

  // Import DisplayMode enum from shader header (shared between C++ and Slang)
  using Mode = shaderio::imcmp::DisplayMode;

  // Import FLIPMode enum from shader header (shared between C++ and Slang)
  using FLIPMode = shaderio::imcmp::FLIPMode;

  // Struct containing visualization image information
  struct ImageInfo
  {
    VkImage    image;
    VkFormat   format;
    VkExtent2D size;
  };

  // Parameters controlling comparison behavior
  struct Parameters
  {
    bool     enabled           = false;              // Is comparison mode active
    float    splitPosition     = 0.5f;               // Split position [0.0, 1.0]
    Mode     leftSide          = Mode::eCapture;     // What to display on left
    Mode     rightSide         = Mode::eCurrent;     // What to display on right
    float    differenceAmplify = 5.0f;               // Amplification factor
    bool     computeMetrics    = false;              // Compute MSE/PSNR
    FLIPMode flipMode          = FLIPMode::eApprox;  // FLIP quality mode (default: Approx for speed)
  };

  // Resources needed for initialization
  struct Resources
  {
    VkDevice                 device;
    nvvk::ResourceAllocator* allocator;
    VkSampler                sampler;        // Linear sampler for scaling
    nvvk::ProfilerGpuTimer*  profiler;       // Optional GPU profiler
    nvslang::SlangCompiler*  slangCompiler;  // Shared shader compiler
    Parameters*              parameters;     // Pointer to comparison parameters
  };

  // ============================================================================
  // Lifecycle Management
  // ============================================================================

  ImageCompare();
  ~ImageCompare();

  void init(const Resources& res);
  void deinit();

  // ============================================================================
  // Per-Frame Operations
  // ============================================================================

  // Capture the current frame
  void capture(VkCommandBuffer cmd, const ImageInfo& imageInfo);

  // Render comparison view (composites captured + current with split)
  // imageInfo: current render image information
  // outputImageView: where to write the composite result (e.g., COLOR_COMPARISON_OUTPUT from GBuffers)
  // skipMetricsUpdate: if true, skip metrics computation (e.g., when temporal sampling converged)
  void render(VkCommandBuffer cmd, const ImageInfo& imageInfo, VkExtent2D outputSize, VkImageView outputImageView, bool skipMetricsUpdate = false);

  // ============================================================================
  // Resource Management
  // ============================================================================

  // Handle viewport resize - invalidates capture if size changes
  void resize(VkExtent2D newSize);

  // Release the captured image
  void releaseCaptureImage();

  // Rebuild pipelines after shader recompilation (compiles shaders internally)
  void rebuildPipelines();

  // ============================================================================
  // State Queries
  // ============================================================================

  // Check if comparison mode is ready (has valid capture)
  bool hasValidCaptureImage() const { return m_hasValidCapture; }

  // Get the composite output image for UI display
  VkImageView getCompositeOutputView() const { return m_compositeOutput.descriptor.imageView; }
  VkImage     getCompositeOutputImage() const { return m_compositeOutput.image; }

  // Get quality metrics (MSE/PSNR/FLIP)
  float getMSE() const { return m_mseValue; }
  float getPSNR() const { return m_psnrValue; }
  float getFLIP() const { return m_flipValue; }
  bool  hasMetricsResult() const { return m_hasMetricsResult; }

  // Collect metrics result from previous frame (called by GaussianSplatting each frame)
  void collectMetricsResult();

  // Configure metrics history buffer size (1 = no graph, N = store last N samples)
  void setMetricsHistorySize(int size);

private:
  // ============================================================================
  // Internal Implementation
  // ============================================================================

  // Shader compilation helper
  bool compileSlangShader(const std::string& filename, VkShaderModule& module);

  // Pipeline initialization
  void initCompositePipeline();
  void deinitCompositePipeline();
  void initCompositeDescriptorSet();
  void updateCompositeDescriptorSet(VkImageView currentImageView, VkImageView outputImageView);

  void initMetricsPipeline();
  void deinitMetricsPipeline();
  void initMetricsDescriptorSet();
  void updateMetricsDescriptorSet(VkImageView currentImageView);

  // Rendering operations

  void updateCurrentImageIfNeeded(const ImageInfo& imageInfo);

  void renderComposite(VkCommandBuffer cmd, VkExtent2D outputSize, VkImageView outputImageView);

  void computeMetrics(VkCommandBuffer cmd, VkExtent2D outputSize);

  void readBackMetricsResult(VkCommandBuffer cmd);

  // ============================================================================
  // Member Variables
  // ============================================================================

  // Init resources
  VkDevice                 m_device        = VK_NULL_HANDLE;
  nvvk::ResourceAllocator* m_alloc         = nullptr;
  VkSampler                m_sampler       = VK_NULL_HANDLE;
  nvvk::ProfilerGpuTimer*  m_profiler      = nullptr;
  nvslang::SlangCompiler*  m_slangCompiler = nullptr;
  Parameters*              m_params        = nullptr;

  // Capture image storage
  nvvk::Image m_captureImage;
  VkExtent2D  m_captureSize{};
  VkFormat    m_captureFormat   = VK_FORMAT_UNDEFINED;
  bool        m_hasValidCapture = false;

  // Current image (set each frame, not owned)
  VkImage     m_currentImage     = VK_NULL_HANDLE;
  VkImageView m_currentImageView = VK_NULL_HANDLE;
  VkExtent2D  m_currentSize{};
  VkFormat    m_currentFormat = VK_FORMAT_UNDEFINED;

  // Composite output (owned by this class)
  nvvk::Image m_compositeOutput;

  // Comparison composite pipeline
  VkPipeline            m_compositePipeline       = VK_NULL_HANDLE;
  VkPipelineLayout      m_compositePipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_compositeDescSetLayout  = VK_NULL_HANDLE;
  VkDescriptorSet       m_compositeDescSet        = VK_NULL_HANDLE;
  VkDescriptorPool      m_compositeDescPool       = VK_NULL_HANDLE;
  VkShaderModule        m_compositeShader         = VK_NULL_HANDLE;

  // MSE/PSNR computation pipeline
  VkPipeline            m_metricsPipeline       = VK_NULL_HANDLE;
  VkPipelineLayout      m_metricsPipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_metricsDescSetLayout  = VK_NULL_HANDLE;
  VkDescriptorSet       m_metricsDescSet        = VK_NULL_HANDLE;
  VkDescriptorPool      m_metricsDescPool       = VK_NULL_HANDLE;
  VkShaderModule        m_metricsShader         = VK_NULL_HANDLE;

  // Metrics buffers
  nvvk::Buffer m_metricsResultDevice;
  nvvk::Buffer m_metricsResultHost;
  float        m_mseValue         = 0.0f;
  float        m_psnrValue        = 0.0f;
  float        m_flipValue        = 0.0f;
  bool         m_hasMetricsResult = false;
  bool         m_canCollectResult = false;

  // Metrics history (circular buffer for graphing)
  std::vector<float> m_mseHistory;
  std::vector<float> m_psnrHistory;
  std::vector<float> m_flipHistory;
  int                m_historySize        = 1;  // Buffer size (1 = no graph)
  int                m_historyIndex       = 0;  // Current write position
  int                m_historySampleCount = 0;  // Number of samples written (capped at historySize)

  // Lazy update flag
  bool m_descriptorNeedsUpdate = false;
};

}  // namespace vk_gaussian_splatting
