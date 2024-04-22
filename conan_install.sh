#!/bin/bash

conan install . -s build_type=Release
conan install . --build=missing -s build_type=Debug