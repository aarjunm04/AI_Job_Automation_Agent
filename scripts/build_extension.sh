#!/bin/bash
# =============================================================================
# AI JOB AUTOMATION AGENT - CHROME EXTENSION BUILD SCRIPT
# =============================================================================
# Builds and packages the Chrome extension for distribution
# =============================================================================

set -e  # Exit on any error

echo "🔨 Building AI Job Automation Chrome Extension..."

# Configuration
EXTENSION_DIR="chrome_extension"
BUILD_DIR="build"
DIST_DIR="dist"
VERSION=$(cat ${EXTENSION_DIR}/manifest.json | grep '"version"' | cut -d'"' -f4)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Extension Version: ${VERSION}${NC}"

# Create build and dist directories
echo -e "${YELLOW}📁 Creating build directories...${NC}"
rm -rf ${BUILD_DIR} ${DIST_DIR}
mkdir -p ${BUILD_DIR} ${DIST_DIR}

# Copy extension files
echo -e "${YELLOW}📋 Copying extension files...${NC}"
cp -r ${EXTENSION_DIR}/* ${BUILD_DIR}/

# Validate manifest.json
echo -e "${YELLOW}✅ Validating manifest.json...${NC}"
if ! python3 -m json.tool ${BUILD_DIR}/manifest.json > /dev/null; then
    echo -e "${RED}❌ Invalid manifest.json${NC}"
    exit 1
fi

# Generate missing icons if they don't exist
echo -e "${YELLOW}🎨 Checking extension icons...${NC}"
if [ ! -f "${BUILD_DIR}/icons/icon-16.png" ]; then
    echo -e "${YELLOW}📐 Generating missing icons...${NC}"
    
    # Create simple gradient icons using ImageMagick (if available)
    if command -v convert &> /dev/null; then
        for size in 16 32 48 128; do
            convert -size ${size}x${size} \
                gradient:'#667eea-#764ba2' \
                -font Arial -pointsize $((size/2)) \
                -fill white -gravity center \
                -annotate +0+0 '🤖' \
                ${BUILD_DIR}/icons/icon-${size}.png
        done
        echo -e "${GREEN}✅ Icons generated${NC}"
    else
        echo -e "${YELLOW}⚠️  ImageMagick not found, creating placeholder icons${NC}"
        for size in 16 32 48 128; do
            # Create a simple colored square as placeholder
            echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77wgAAAABJRU5ErkJggg==" | base64 -d > ${BUILD_DIR}/icons/icon-${size}.png
        done
    fi
fi

# Minify JavaScript files (if uglify-js is available)
echo -e "${YELLOW}⚡ Optimizing JavaScript files...${NC}"
if command -v uglifyjs &> /dev/null; then
    echo -e "${YELLOW}📦 Minifying JavaScript...${NC}"
    
    # Minify background script
    if [ -f "${BUILD_DIR}/background.js" ]; then
        uglifyjs ${BUILD_DIR}/background.js -o ${BUILD_DIR}/background.min.js -c -m
        mv ${BUILD_DIR}/background.min.js ${BUILD_DIR}/background.js
        echo -e "${GREEN}✅ background.js minified${NC}"
    fi
    
    # Minify content script
    if [ -f "${BUILD_DIR}/content/content.js" ]; then
        uglifyjs ${BUILD_DIR}/content/content.js -o ${BUILD_DIR}/content/content.min.js -c -m
        mv ${BUILD_DIR}/content/content.min.js ${BUILD_DIR}/content/content.js
        echo -e "${GREEN}✅ content.js minified${NC}"
    fi
    
    # Minify sidebar script
    if [ -f "${BUILD_DIR}/sidebar/sidebar.js" ]; then
        uglifyjs ${BUILD_DIR}/sidebar/sidebar.js -o ${BUILD_DIR}/sidebar/sidebar.min.js -c -m
        mv ${BUILD_DIR}/sidebar/sidebar.min.js ${BUILD_DIR}/sidebar/sidebar.js
        echo -e "${GREEN}✅ sidebar.js minified${NC}"
    fi
    
    # Minify utility scripts
    for file in ${BUILD_DIR}/utils/*.js; do
        if [ -f "$file" ]; then
            filename=$(basename "$file" .js)
            uglifyjs "$file" -o "${BUILD_DIR}/utils/${filename}.min.js" -c -m
            mv "${BUILD_DIR}/utils/${filename}.min.js" "$file"
            echo -e "${GREEN}✅ ${filename}.js minified${NC}"
        fi
    done
else
    echo -e "${YELLOW}⚠️  uglify-js not found, skipping minification${NC}"
fi

# Minify CSS files (if cleancss is available)
if command -v cleancss &> /dev/null; then
    echo -e "${YELLOW}🎨 Minifying CSS files...${NC}"
    
    if [ -f "${BUILD_DIR}/sidebar/sidebar.css" ]; then
        cleancss -o ${BUILD_DIR}/sidebar/sidebar.min.css ${BUILD_DIR}/sidebar/sidebar.css
        mv ${BUILD_DIR}/sidebar/sidebar.min.css ${BUILD_DIR}/sidebar/sidebar.css
        echo -e "${GREEN}✅ sidebar.css minified${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  clean-css not found, skipping CSS minification${NC}"
fi

# Remove development files
echo -e "${YELLOW}🧹 Cleaning development files...${NC}"
find ${BUILD_DIR} -name "*.md" -delete
find ${BUILD_DIR} -name "*.txt" -delete
find ${BUILD_DIR} -name ".DS_Store" -delete
find ${BUILD_DIR} -name "Thumbs.db" -delete

# Validate file sizes
echo -e "${YELLOW}📏 Checking file sizes...${NC}"
total_size=$(du -sh ${BUILD_DIR} | cut -f1)
echo -e "${BLUE}Total extension size: ${total_size}${NC}"

# Check for large files
large_files=$(find ${BUILD_DIR} -type f -size +1M)
if [ ! -z "$large_files" ]; then
    echo -e "${YELLOW}⚠️  Large files found (>1MB):${NC}"
    echo "$large_files"
fi

# Create development version (unpackaged)
echo -e "${YELLOW}📦 Creating development build...${NC}"
cp -r ${BUILD_DIR} ${DIST_DIR}/ai-job-automation-dev-v${VERSION}

# Create production ZIP package
echo -e "${YELLOW}🗜️  Creating production package...${NC}"
cd ${BUILD_DIR}
zip -r ../${DIST_DIR}/ai-job-automation-v${VERSION}.zip . -x "*.map" "*.log"
cd ..

# Create CRX package (if crx3 is available)
if command -v crx3 &> /dev/null; then
    echo -e "${YELLOW}🔐 Creating CRX package...${NC}"
    
    # Generate private key if it doesn't exist
    if [ ! -f "extension-key.pem" ]; then
        echo -e "${YELLOW}🔑 Generating extension private key...${NC}"
        openssl genrsa -out extension-key.pem 2048
    fi
    
    # Create CRX file
    crx3 ${BUILD_DIR} -o ${DIST_DIR}/ai-job-automation-v${VERSION}.crx -p extension-key.pem
    echo -e "${GREEN}✅ CRX package created${NC}"
else
    echo -e "${YELLOW}⚠️  crx3 not found, skipping CRX creation${NC}"
fi

# Generate checksums
echo -e "${YELLOW}🔍 Generating checksums...${NC}"
cd ${DIST_DIR}
if command -v sha256sum &> /dev/null; then
    sha256sum *.zip *.crx > checksums.sha256 2>/dev/null || true
elif command -v shasum &> /dev/null; then
    shasum -a 256 *.zip *.crx > checksums.sha256 2>/dev/null || true
fi
cd ..

# Create update manifest
echo -e "${YELLOW}📄 Creating update manifest...${NC}"
cat > ${DIST_DIR}/update_manifest.json << EOF
{
  "addons": {
    "ai-job-automation@extension": {
      "updates": [
        {
          "version": "${VERSION}",
          "update_link": "https://github.com/your-repo/ai-job-automation/releases/download/v${VERSION}/ai-job-automation-v${VERSION}.crx"
        }
      ]
    }
  }
}
EOF

# Generate installation instructions
echo -e "${YELLOW}📖 Creating installation instructions...${NC}"
cat > ${DIST_DIR}/INSTALL.md << EOF
# AI Job Automation Chrome Extension - Installation Guide

## Version: ${VERSION}
## Build Date: $(date)

### Development Installation (Recommended for testing)

1. Download and extract: \`ai-job-automation-dev-v${VERSION}\`
2. Open Chrome and navigate to \`chrome://extensions/\`
3. Enable "Developer mode" (top right toggle)
4. Click "Load unpacked"
5. Select the extracted \`ai-job-automation-dev-v${VERSION}\` folder
6. The extension will be installed and ready to use

### Production Installation (ZIP package)

1. Download: \`ai-job-automation-v${VERSION}.zip\`
2. Extract the ZIP file
3. Follow steps 2-6 from Development Installation above

### Store Installation (CRX package)

1. Download: \`ai-job-automation-v${VERSION}.crx\`
2. Open Chrome and navigate to \`chrome://extensions/\`
3. Enable "Developer mode"
4. Drag and drop the CRX file onto the extensions page
5. Click "Add extension" when prompted

### Configuration

1. Click the extension icon in the Chrome toolbar
2. Configure your API keys in the Settings tab:
   - OpenAI API Key (required)
   - Perplexity API Key (optional, for company research)
   - Notion API Key (required for job tracking)
3. Set up your user profile information
4. Navigate to any job site and start using the AI features!

### Supported Job Sites

- LinkedIn
- Indeed
- Naukri
- FlexJobs
- WeWorkRemotely
- RemoteOK
- Wellfound (AngelList)
- Glassdoor
- And 12+ more platforms

### Features

- 🧠 AI-powered job analysis with match scoring
- 📝 Automatic form filling
- 🔍 Real-time company research
- 📄 Resume optimization recommendations
- 📊 Job application tracking
- 🚀 Quick apply functionality

### Troubleshooting

If you encounter issues:

1. Check that all API keys are properly configured
2. Ensure the backend system is running (if using local setup)
3. Verify that the extension has necessary permissions
4. Check the browser console for error messages
5. Try disabling and re-enabling the extension

### Support

For support and updates, visit: https://github.com/your-repo/ai-job-automation
EOF

# Final summary
echo -e "\n${GREEN}🎉 Build completed successfully!${NC}"
echo -e "${BLUE}📦 Build artifacts:${NC}"
ls -la ${DIST_DIR}/
echo -e "\n${GREEN}✅ Ready for distribution!${NC}"

# Development testing suggestion
echo -e "\n${YELLOW}💡 For testing:${NC}"
echo -e "1. Load '${DIST_DIR}/ai-job-automation-dev-v${VERSION}' as unpacked extension"
echo -e "2. Configure API keys in extension settings"
echo -e "3. Visit a job site and test the features"

echo -e "\n${BLUE}🚀 Build process completed at $(date)${NC}"
