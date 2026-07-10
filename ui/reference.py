"""Technical Report and References tabs — static documentation renderers."""
import streamlit as st


def render_technical_report():
    st.header("Technical Report: Generative Mix Design")
    with open("docs/TECHNICAL_REPORT.md", "r", encoding="utf-8") as f:
        report_content = f.read()
    st.markdown(report_content)


def render_references():
    st.header("References & Resources")

    st.subheader("Software & Libraries")
    st.markdown("""
    | Component | Library | Version | Link |
    |-----------|---------|---------|------|
    | ML Prediction | XGBoost | 3.x | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
    | Bayesian Inference | BayesFlow | 1.x | [github.com/stefanradev93/BayesFlow](https://github.com/stefanradev93/BayesFlow) |
    | Deep Learning | TensorFlow | 2.14.* | [tensorflow.org](https://www.tensorflow.org/) |
    | Probabilistic | TensorFlow Probability | 0.22.* | [tensorflow.org/probability](https://www.tensorflow.org/probability) |
    | Visualization | Plotly | 5.18+ | [plotly.com/python](https://plotly.com/python/) |
    | Web Framework | Streamlit | 1.30+ | [streamlit.io](https://streamlit.io/) |
    | Scientific | NumPy, SciPy, Pandas | — | [numpy.org](https://numpy.org/) |
    """)

    st.subheader("Primary Research References")
    st.markdown("""
    **Machine Learning & Concrete Prediction**

    1. **Yeh, I-C. (1998).** "Modeling of strength of high-performance concrete using artificial neural networks."
       *Cement and Concrete Research*, 28(12), 1797-1808.
       *[The foundational dataset for this project. Established that nonlinear models outperform traditional regression for concrete.]*

    2. **Chen, T. & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System."
       *Proceedings of the 22nd ACM SIGKDD*, 785-794.
       *[The algorithm behind our forward predictor. Gradient boosting with regularization.]*

    3. **DeRousseau, M.A. et al. (2019).** "A comparison of machine learning methods for predicting compressive strength of concrete."
       *Construction and Building Materials*, 161, 164-176.
       *[Benchmark study comparing RF, SVR, ANN, and XGBoost on concrete datasets.]*

    **Bayesian & Amortized Inference**

    4. **Radev, S.T. et al. (2020).** "BayesFlow: Learning complex stochastic models with invertible neural networks."
       *IEEE Transactions on Neural Networks and Learning Systems*, 31(11), 5051-5064.
       *[The theoretical foundation for our inverse design engine.]*

    5. **Papamakarios, G. et al. (2021).** "Normalizing Flows for Probabilistic Modeling and Inference."
       *Journal of Machine Learning Research*, 22(57), 1-64.
       *[Comprehensive review of the normalizing flow architecture we use for posterior estimation.]*

    6. **Cranmer, K. et al. (2020).** "The frontier of simulation-based inference."
       *PNAS*, 117(48), 30055-30062.
       *[Contextualizes amortized inference within broader scientific simulation.]*

    **Cement Chemistry**

    7. **Bogue, R.H. (1929).** "Calculation of the Compounds in Portland Cement."
       *Industrial & Engineering Chemistry Analytical Edition*, 1(4), 192-197.
       *[The classic calculation for estimating clinker phases from oxide composition.]*

    8. **Taylor, H.F.W. (1997).** *Cement Chemistry*, 2nd Edition. Thomas Telford.
       *[The definitive textbook on cement hydration—our Tier 2 chemistry model is inspired by chapters 6-8.]*

    9. **Lothenbach, B., Scrivener, K., & Hooton, R.D. (2011).** "Supplementary cementitious materials."
       *Cement and Concrete Research*, 41(12), 1244-1256.
       *[Comprehensive review of SCM chemistry that informs our pozzolanic reaction model.]*

    10. **Scrivener, K.L. et al. (2015).** "TC 238-SCM: Hydration and microstructure of concrete with SCMs."
        *Materials and Structures*, 48, 835-862.
        *[State-of-the-art on SCM hydration mechanisms.]*

    **Sustainability & Carbon Accounting**

    11. **WBCSD/CSI (2013).** "The Cement CO2 and Energy Protocol: CO2 and Energy Accounting
        and Reporting Standard for the Cement Industry." World Business Council for Sustainable Development.
        *[Industry-standard methodology for carbon accounting that our chemistry layer follows.]*

    12. **Habert, G. et al. (2020).** "Environmental impacts and decarbonization strategies in the cement industry."
        *Nature Reviews Earth & Environment*, 1, 559-573.
        *[Modern perspective on cement decarbonization—motivates our multi-objective optimization.]*

    **Optimization & Metaheuristics**

    13. **Holland, J.H. (1975).** *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
        *[The original text on genetic algorithms.]*

    14. **Kirkpatrick, S., Gelatt, C.D., & Vecchi, M.P. (1983).** "Optimization by Simulated Annealing."
        *Science*, 220(4598), 671-680.
        *[The foundational paper on simulated annealing—our SA implementation follows this formulation.]*

    15. **Deb, K. (2001).** *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley.
        *[Theoretical framework for Pareto optimization; explains scalarization vs. Pareto dominance.]*
    """)

    st.subheader("Further Reading")
    st.markdown("""
    - **Neville, A.M. (2011).** *Properties of Concrete*, 5th Edition. Pearson.
      [Comprehensive reference on concrete materials and behavior]

    - **Mehta, P.K. & Monteiro, P.J.M. (2014).** *Concrete: Microstructure, Properties, and Materials*, 4th Edition. McGraw-Hill.
      [Authoritative textbook covering concrete from microstructure to durability]

    - **ACI 211.1-91.** "Standard Practice for Selecting Proportions for Normal, Heavyweight, and Mass Concrete."
      [Traditional mix design methodology]
    """)

    st.subheader("Acknowledgments")
    st.markdown("""
    This tool was developed to accelerate sustainable concrete design and democratize access to advanced
    optimization techniques. We acknowledge the researchers who made their datasets publicly available,
    the open-source community (NumPy, SciPy, TensorFlow, Streamlit, Plotly), and the cement science
    community whose decades of research make computational models possible.

    Special thanks to the UCI Machine Learning Repository for hosting the Concrete Compressive Strength dataset.
    """)
