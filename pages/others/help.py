import streamlit as st

st.title("Besoin d'aide ?")

# Introduction
st.markdown("""
## Foire Aux Questions (FAQ)

Nous avons rassemblé les réponses aux questions les plus courantes concernant nos outils de simulation de prêt.  
Si vous ne trouvez pas ce que vous cherchez, veuillez visiter notre page [Contact](contact) pour poser votre question.
""")

# FAQ Accordion
with st.expander("Comment utiliser le simulateur de prêt basique ?"):
    st.markdown("""
    Le simulateur de prêt basique vous permet de calculer vos mensualités et les intérêts totaux pour un prêt simple.

    1. Saisissez le montant du prêt
    2. Indiquez le taux d'intérêt
    3. Définissez la durée du prêt en années
    4. Cliquez sur "Calculer" pour voir les résultats

    Le simulateur affichera le montant de la mensualité, le coût total du prêt et les intérêts totaux payés.
    """)

with st.expander("Quelle est la différence entre le simulateur de prêt basique et avancé ?"):
    st.markdown("""
    **Simulateur de prêt basique :**
    - Entrées simples : montant du prêt, taux d’intérêt, durée
    - Calcule les mensualités et les intérêts totaux
    - Idéal pour les prêts classiques

    **Simulateur de prêt avancé :**
    - Fonctionnalités supplémentaires : apport personnel, assurance, taxes
    - Taux d’intérêt variables
    - Tableau d’amortissement détaillé
    - Calculs avec paiements anticipés
    - Idéal pour les prêts immobiliers et les financements complexes
    """)

with st.expander("Quelle est la précision des simulations de prêt ?"):
    st.markdown("""
    Nos simulateurs utilisent des formules financières standard pour calculer les mensualités et les intérêts.  
    Cependant, les conditions réelles de votre prêt peuvent varier en fonction de :

    - Votre score de crédit
    - Les politiques spécifiques de votre banque ou prêteur
    - Les frais supplémentaires éventuels
    - Des arrondis appliqués aux calculs

    Nos outils sont conçus pour la planification, mais nous vous recommandons de consulter un conseiller financier pour toute décision finale.
    """)

with st.expander("Puis-je sauvegarder ou exporter mes résultats de simulation ?"):
    st.markdown("""
    Oui ! Chaque simulateur propose une option d'exportation permettant de :

    - Télécharger les résultats sous format CSV
    - Sauvegarder les tableaux d’amortissement sous format Excel
    - Générer un résumé en PDF

    Recherchez les boutons de téléchargement sous les résultats de la simulation.
    """)

with st.expander("J’ai trouvé un bug ou j’ai une suggestion de fonctionnalité"):
    st.markdown("""
    Nous améliorons constamment nos outils grâce aux retours des utilisateurs. Si vous avez identifié un bug ou avez une idée de nouvelle fonctionnalité :

    1. Rendez-vous sur notre page [Contact](contact)
    2. Décrivez le problème ou votre suggestion en détail
    3. Incluez les étapes pour reproduire un éventuel bug
    4. Soumettez votre demande

    Notre équipe de développement analyse tous les retours et priorise les mises à jour en fonction des besoins des utilisateurs.
    """)

# Support par email
st.header("Support par Email")

st.markdown("""
Si votre question n'est pas couverte dans notre FAQ, nous offrons un support par email détaillé. Voici notre processus :
""")

st.info("""
### Notre Processus de Support par Email

1. **Soumettez votre question** via notre page [Contact](contact)

2. **Recevez une confirmation** immédiate de la prise en compte de votre demande

3. **Analyse par un expert** - Votre question est attribuée au spécialiste le plus qualifié de notre équipe

4. **Recherche approfondie** - Nous prenons le temps d’examiner votre situation avec précision

5. **Réponse détaillée** - Sous 1 à 2 jours ouvrés, vous recevrez un email complet répondant à vos questions

6. **Suivi personnalisé** - Si notre réponse nécessite des précisions ou soulève de nouvelles interrogations, il vous suffit de répondre pour continuer la discussion

Nous privilégions des réponses réfléchies et précises plutôt que des réponses rapides mais incomplètes. Ainsi, nous vous garantissons des informations fiables et adaptées à votre situation.
""")

# Tutoriels vidéo
st.header("Tutoriels Vidéo")

st.markdown("""
Notre bibliothèque de tutoriels vidéo vous guide pas à pas dans l'utilisation de nos outils de simulation de prêt.

- [Tutoriel du simulateur de prêt basique](#)
- [Présentation des fonctionnalités avancées](#)
- [Comparaison de plusieurs options de prêt](#)
- [Création de scénarios de prêt](#)

*(Remarque : Dans une version réelle, ces liens pointeraient vers des vidéos explicatives)*
""")

# Invitation à contacter le support
st.markdown("""
## Vous avez encore des questions ?

Si vous ne trouvez pas les réponses que vous cherchez ici, n'hésitez pas à nous [contacter](contact).  
Notre équipe est à votre disposition pour vous fournir une assistance personnalisée pour toutes vos simulations de prêt.
""")
