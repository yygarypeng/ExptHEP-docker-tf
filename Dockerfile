# LICENSE: MIT (Gary Peng: Dec. 09, 2023)
# This dockerfile build on the Ubuntu 20.04.5 LTS (focal); the base server env is from tensorflow official image (tensorflow/tensorlfow:2.13.0-gpu)

FROM tensorflow/tensorflow:2.13.0-gpu

LABEL maintainer="Gary_Peng <yygarypeng@gapp.nthu.edu.tw>"

ENV SUDO_FORCE_REMOVE=yes
ENV DEBIAN_FRONTEND=noninteractive

RUN buildDeps="git vim wget openssl htop glances libgl1-mesa-glx" && \
	apt-get update && \
	apt-get install -y $buildDeps && \
	apt-get clean

RUN mkdir -p ~/miniconda3 && \
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
	bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
	rm -rf ~/miniconda3/miniconda.sh
ENV PATH=/root/miniconda3/bin:$PATH
RUN conda init

ADD tf.yml /root/miniconda3/tf.yml 
RUN conda env create --name tf -f /root/miniconda3/tf.yml

RUN mkdir ~/work/ && \
	mkdir ~/data/

RUN echo "parse_git_branch() {\n git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \\(.*\\)/ (\\1)/'\n}" >> ~/.bashrc && \
    echo "export PS1='\[\033[32m\]\w\[\033[33m\]\$(parse_git_branch)\[\033[00m\] $ '\n" >> ~/.bashrc && \
    echo "# Other alias\n\
alias lesf='less +F'\n\
alias lesg='less +G'\n\
alias cc='clear'\n\
alias data='cd ~/data/'\n\
alias work='cd ~/work/'\n\
alias home='cd ~'\n\
\n\
bind 'set show-all-if-ambiguous on'\n\
bind 'set completion-ignore-case on'\n\
bind 'TAB:menu-complete'\n\
" >> ~/.bashrc

# Use a separate RUN command for conda activate and cd ~/work/
RUN echo "conda activate tf" >> ~/.bashrc && \
    echo "cd ~/work/" >> ~/.bashrc

# Add test code in home directory
ADD test.py /root/test.py

CMD ["/bin/bash"]
