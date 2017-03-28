brew install llvm
ln -s /usr/local/opt/llvm/bin/clang /usr/local/bin/clang-omp

brew unlink gcc
brew install gcc --without-multilib 

brew install homebrew/science/<>
