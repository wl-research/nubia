from setuptools import setup

setup(
    name='nubia_score',
    version='0.1.5',
    description='NUBIA (NeUral Based Interchangeability Assessor) is a SoTA evaluation metric for text generation',
    url='https://github.com/wl-research/nubia',
    author='Ali Abdalla',
    author_email='ali.si3luwa@gmail.com',
    include_package_data=True,
    license='MIT License',
    packages=['nubia_score'],
    keywords=['machine learning', 'nlp', 'text', 'evaluation', 'metrics'],
    install_requires=[
        'torch>=1.4.0',
        'fairseq',
        'pytorch-transformers',
        'numpy>=1.18.2',
        'joblib',
        'scikit-learn>=0.22.2',
        'wget',
    ],
)