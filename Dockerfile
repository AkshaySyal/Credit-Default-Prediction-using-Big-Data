FROM ubuntu:22.04

# Install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y wget && \
    apt-get install -y curl && \
    apt-get install -y make
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y maven && \
    apt-get install -y awscli 
    
# Install scala 2.12 (package manager would install 2.11)
RUN curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup -y
ENV PATH="$PATH:/root/.local/share/coursier/bin"
RUN cs install scala:2.12.17 && cs install scalac:2.12.17

# Download hadoop
RUN wget https://downloads.apache.org/hadoop/common/hadoop-3.3.5/hadoop-3.3.5.tar.gz && \
    tar -xvzf hadoop-3.3.5.tar.gz && \
    mv hadoop-3.3.5 /usr/local/hadoop-3.3.5

# Download spark
RUN wget https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-without-hadoop.tgz && \
    tar -xvzf spark-3.3.2-bin-without-hadoop.tgz && \
    mv spark-3.3.2-bin-without-hadoop /usr/local/spark-3.3.2-bin-without-hadoop

# Set environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV HADOOP_HOME=/usr/local/hadoop-3.3.5
ENV YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV SCALA_HOME=/usr/share/scala
ENV SPARK_HOME=/usr/local/spark-3.3.2-bin-without-hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SCALA_HOME/bin:$SPARK_HOME/bin
RUN echo "export SPARK_DIST_CLASSPATH=$(hadoop classpath)" >> ~/.bash_aliases
# ENV SPARK_DIST_CLASSPATH=$(hadoop classpath)
RUN echo JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 >> /usr/local/hadoop-3.3.5/etc/hadoop/hadoop-env.sh

# Set working directory
ADD . /app/
WORKDIR /app/

