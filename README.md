# Enterprise-Level-ImageGenerator

## 🌟 Core Functionality
- Multi-Provider Support : Integrates with Leonardo AI, Ideogram, and Google's Imagen-4 for image generation
- Advanced Prompt Enhancement : Uses Qwen AI to enhance prompts with ethnicity, activities, facial expressions, and fur colors
- Background Processing : Automatic background removal using Photoroom and BiRefNet
- Card Template Application : Applies generated images to customizable card templates
- Batch Processing : Handles multiple images and ZIP file inputs
## Key Components
### Image Generation Pipeline
- Provider Selection : Supports Leonardo AI (Flux Dev, Phoenix), Ideogram (V2/V3), and Imagen-4 (Fast/ Regular/Ultra)
- Reference Image Support : Multi-reference system with different types (style, character, content)
- Prompt Modification : Dynamic enhancement with ethnic traits, activities, expressions, and fur colors
- Quality Control : Comprehensive validation and error handling
### Processing Features
- Dual Output System : Creates both white background and card template versions
- Metadata Generation : Creates detailed Excel reports with embedded images
- ZIP Archive Creation : Comprehensive packaging with proper naming conventions
- Base64 Encoding : Optional encoding for web integration
### Cloud Integration
- S3 Upload : Automated upload to AWS S3 with organized folder structure
- Google Drive Integration : Hierarchical folder organization with theme/category/subcategory structure
- Batch Upload Support : Handles multiple files with progress tracking
### Advanced Features
- Ethnic Enhancement : Detailed trait mapping for various ethnicities (Javanese, Sundanese, Chinese Indonesian, American, Hindi)
- Gender Detection : Automatic gender identification and enhancement in prompts
- Activity Generation : Dynamic activity and expression suggestions
- Watermark Removal : Intelligent watermark detection and removal
- Filename Conventions : Structured naming using theme and category codes
### User Interface
- Gradio-based UI : Interactive web interface with multiple tabs
- Real-time Processing : Live status updates and progress tracking
- Gallery Display : Organized image viewing with metadata
- Download Management : Multiple download options for different output types
## Technical Architecture
- Modular Design : Well-organized functions for different processing stages
- Error Handling : Comprehensive exception management and logging
- Configuration Management : Extensive mapping dictionaries for themes, categories, and providers
- Async Support : Handles both synchronous and asynchronous operations
