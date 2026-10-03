project(
    'neural-network-cpp',
    'cpp',
    version: '0.1.0',
    meson_version: '>=1.3.0',
    default_options: [
        'warning_level=3',
        'cpp_std=c++20',
    ],
)

inc = include_directories('include')

sources = files(
    'src/main.cpp',
    'src/Node.cpp',
)

executable('nn-out', sources, include_directories: inc)
