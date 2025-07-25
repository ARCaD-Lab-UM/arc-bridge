#!/bin/bash

# Detect OS type
OS_TYPE=$(uname -s)
if [[ "$OS_TYPE" == "Darwin" ]]; then
    SYSTEM="macOS"
    MATLAB_PREF_DIR_BASE=~/Library/Application\ Support/MathWorks/MATLAB
elif [[ "$OS_TYPE" == "Linux" ]]; then
    SYSTEM="Linux"
    MATLAB_PREF_DIR_BASE=~/.matlab
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

echo "Detected OS: $SYSTEM"

LCM_TYPE_PATH=lcm_types
LCM_GEN_DES_PATH=arc_bridge

LCM_PKG_DIR=lcm_msgs
JAVA_ARCHIVE_NAME=lcm_msgs.jar

LCM_PKG_PATH=$LCM_GEN_DES_PATH/$LCM_PKG_DIR
LCM_JAVA_ARCHIVE_PATH=$LCM_PKG_PATH/$JAVA_ARCHIVE_NAME

# Try to automatically determine where the LCM java file is
LCM_JAR_PATH=`pkg-config --variable=classpath lcm-java`
if [ $? != 0 ] ; then
    if [ -e /usr/local/share/java/lcm.jar ] ; then
        LCM_JAR_PATH=/usr/local/share/java/lcm.jar
    else
        if [ -e $LCM_TYPE_PATH/lcm/lcm-java/lcm.jar ] ; then
            LCM_JAR_PATH=$LCM_TYPE_PATH/lcm/lcm-java/lcm.jar
        else
            LCM_JAR_PATH=$LCM_TYPE_PATH/lcm.jar
        fi
    fi
fi
echo "Found LCM jar at path: $LCM_JAR_PATH"

# Remove old generated types
echo "Removing old generated types from $LCM_PKG_PATH"
rm -r $LCM_PKG_PATH

# Make a folder to store generated types if not exists
mkdir -p $LCM_GEN_DES_PATH

echo "Generating LCM types from $LCM_TYPE_PATH/*.lcm"
# Generate python types
lcm-gen -p $LCM_TYPE_PATH/*.lcm --ppath $LCM_GEN_DES_PATH
# Generate java types
lcm-gen -j $LCM_TYPE_PATH/*.lcm --jpath $LCM_GEN_DES_PATH
# Generate C++ types
lcm-gen -x $LCM_TYPE_PATH/*.lcm --cpp-hpath $LCM_GEN_DES_PATH

# Compile java types
echo "Compiling generated Java types"
# java release 8 is used for compatability of MATLAB >= 2023b
javac -cp $LCM_JAR_PATH $LCM_PKG_PATH/*.java --release 8 

cd $LCM_GEN_DES_PATH
jar cf $LCM_PKG_DIR/$JAVA_ARCHIVE_NAME $LCM_PKG_DIR/*.class
cd ..

# Find the latest MATLAB preference folder
if [[ -d "$MATLAB_PREF_DIR_BASE" ]]; then
    MATLAB_LATEST_PREF_DIR=$(ls -d "$MATLAB_PREF_DIR_BASE"/R* 2>/dev/null | sort -V | tail -n 1)
    
    if [[ -z "$MATLAB_LATEST_PREF_DIR" ]]; then
        echo "No MATLAB preference folder found."
        exit 1
    fi

    echo "Using MATLAB preference folder: $MATLAB_LATEST_PREF_DIR"
else
    echo "MATLAB preference base directory not found: $MATLAB_PREF_DIR_BASE"
    exit 1
fi

# Write to javaclasspath.txt
JAVACLASS_PATH_FILE="$MATLAB_LATEST_PREF_DIR/javaclasspath.txt"

# Add the generated LCM types to javaclasspath.txt
echo "Adding generated LCM types to $JAVACLASS_PATH_FILE"
echo -e "$LCM_JAR_PATH\n$LCM_JAVA_ARCHIVE_PATH\n" > "$JAVACLASS_PATH_FILE"
