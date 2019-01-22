from setuptools import setup, find_packages

setup(
    name            = 'recommenderhsw',
    version         = '1.0',
    description     = 'Python Recommender System Package',
    author          = 'Seungwoo Hyun',
    author_email    = 'p829911@gmail.com',
    url             = 'https://github.com/p829911', 
    download_url    = 'https://github.com/p829911/recommenderhsw 
    install_requires= ["numpy", "pandas", "scipy"],
    packages        = find_packages(),
    keywords        = ["recommender"],
    python_requires = '>=3'
)
