# Installation du projet Hawkshot

Ce document explique comment installer **Hawkshot** sur une nouvelle machine Windows, depuis une installation vierge jusqu'à la vérification complète de l'environnement de développement.

> Les commandes de ce guide sont prévues pour **PowerShell** dans Windows 10 ou Windows 11.

---

## 1. Prérequis

Avant de commencer, il faut disposer des éléments suivants :

- un accès à Internet ;
- un compte GitHub ayant accès au dépôt Hawkshot ;
- Visual Studio Code ;
- Git ;
- `uv` ;
- PostgreSQL ;
- éventuellement Power BI Desktop pour la partie visualisation.

Python n'a pas besoin d'être installé manuellement : `uv` peut installer et gérer la version demandée par le projet.

---

## 2. Installer Visual Studio Code

Télécharger puis installer Visual Studio Code :

<https://code.visualstudio.com/>

Pendant l'installation, il est recommandé de cocher :

- **Add "Open with Code" action** ;
- **Add to PATH** ;
- **Register Code as an editor for supported file types**.

Extensions VS Code recommandées :

- Python — Microsoft ;
- Pylance — Microsoft ;
- Ruff — Astral Software ;
- PostgreSQL — Microsoft ou une extension équivalente ;
- GitLens — facultatif ;
- Jupyter — utile si le projet contient des notebooks.

---

## 3. Installer Git

Télécharger Git for Windows :

<https://git-scm.com/download/win>

Pendant l'installation, les options par défaut conviennent généralement.

Vérifier ensuite l'installation dans PowerShell :

```powershell
git --version
```

Configurer son identité Git :

```powershell
git config --global user.name "Prénom Nom"
git config --global user.email "adresse@email.com"
```

Vérifier la configuration :

```powershell
git config --global --list
```

### Authentification GitHub

Lors du premier `git push`, GitHub peut demander une authentification dans le navigateur.

GitHub n'accepte plus le mot de passe du compte pour les opérations Git en HTTPS. Git Credential Manager, installé avec Git for Windows, permet normalement de gérer automatiquement la connexion.

---

## 4. Installer uv

`uv` gère :

- la version de Python ;
- l'environnement virtuel ;
- les dépendances ;
- le fichier `uv.lock` ;
- l'exécution des commandes du projet.

Ouvrir PowerShell puis exécuter :

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Fermer puis rouvrir PowerShell.

Vérifier l'installation :

```powershell
uv --version
```

Mettre `uv` à jour si nécessaire :

```powershell
uv self update
```

---

## 5. Récupérer le dépôt Hawkshot

Choisir un dossier dans lequel stocker les projets, par exemple :

```powershell
mkdir C:\Projets
cd C:\Projets
```

Cloner le dépôt :

```powershell
git clone <URL_DU_DEPOT_HAWKSHOT>
```

Exemple de format :

```powershell
git clone https://github.com/<UTILISATEUR_GITHUB>/Hawkshot.git
```

Entrer dans le projet :

```powershell
cd Hawkshot
```

Ouvrir le projet dans VS Code :

```powershell
code .
```

Vérifier l'état Git :

```powershell
git status
```

---

## 6. Installer Python et les dépendances

Hawkshot utilise **Python 3.11**.

Depuis la racine du projet :

```powershell
uv python install 3.11
```

Installer l'environnement et toutes les dépendances définies dans `pyproject.toml` et `uv.lock` :

```powershell
uv sync
```

Cette commande crée normalement un environnement virtuel dans :

```text
.venv
```

Vérifier la version de Python réellement utilisée :

```powershell
uv run python --version
```

Résultat attendu :

```text
Python 3.11.x
```

### Sélectionner l'interpréteur dans VS Code

Dans VS Code :

1. ouvrir la palette avec `Ctrl + Shift + P` ;
2. rechercher `Python: Select Interpreter` ;
3. sélectionner l'interpréteur situé dans :

```text
Hawkshot\.venv\Scripts\python.exe
```

Il n'est pas obligatoire d'activer manuellement l'environnement virtuel lorsque les commandes sont exécutées avec `uv run`.

Pour l'activer manuellement dans PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

Pour le désactiver :

```powershell
deactivate
```

---

## 7. Installer les hooks pre-commit

Les hooks pre-commit exécutent automatiquement Ruff avant chaque commit.

Installer les hooks Git :

```powershell
uv run pre-commit install
```

Tester les hooks sur tous les fichiers du projet :

```powershell
uv run pre-commit run --all-files
```

Lors du premier lancement, pre-commit peut télécharger son environnement. C'est normal.

Résultat attendu :

```text
ruff-check........................................................Passed
ruff-format.......................................................Passed
```

Si Ruff modifie automatiquement des fichiers, les ajouter de nouveau à Git avant de refaire le commit :

```powershell
git add .
uv run pre-commit run --all-files
```

---

## 8. Vérifier Ruff

Lancer le linter :

```powershell
uv run ruff check .
```

Appliquer automatiquement les corrections disponibles :

```powershell
uv run ruff check . --fix
```

Vérifier le formatage :

```powershell
uv run ruff format --check .
```

Formater les fichiers :

```powershell
uv run ruff format .
```

---

## 9. Vérifier pytest

Lancer tous les tests :

```powershell
uv run pytest
```

Pour obtenir plus de détails :

```powershell
uv run pytest -v
```

Pour arrêter au premier échec :

```powershell
uv run pytest -x
```

Pour exécuter un fichier de test précis :

```powershell
uv run pytest tests\test_nom_du_module.py -v
```

> Toute nouvelle fonction contenant une logique métier importante doit être accompagnée de tests pytest.

---

## 10. Installer PostgreSQL

Télécharger l'installateur Windows depuis :

<https://www.postgresql.org/download/windows/>

Pendant l'installation :

1. conserver PostgreSQL Server ;
2. conserver pgAdmin 4 ;
3. conserver Command Line Tools ;
4. choisir un mot de passe pour l'utilisateur administrateur `postgres` ;
5. conserver le port par défaut `5432`, sauf conflit ;
6. noter soigneusement le mot de passe choisi.

Stack Builder n'est pas indispensable pour Hawkshot et peut être ignoré.

### Vérifier PostgreSQL

Rechercher **SQL Shell (psql)** dans le menu Démarrer.

Les valeurs proposées peuvent généralement être validées avec Entrée :

```text
Server [localhost]:
Database [postgres]:
Port [5432]:
Username [postgres]:
Password for user postgres:
```

Dans `psql`, vérifier la version :

```sql
SELECT version();
```

Quitter `psql` :

```sql
\q
```

---

## 11. Créer la base de données du projet

Se connecter en tant qu'administrateur :

```powershell
psql -U postgres
```

Si la commande `psql` n'est pas reconnue, utiliser SQL Shell ou ajouter le dossier `bin` de PostgreSQL au `PATH`.

Exemple de chemin :

```text
C:\Program Files\PostgreSQL\<VERSION>\bin
```

Créer un utilisateur dédié au projet :

```sql
CREATE USER hawkshot_user WITH PASSWORD 'mot_de_passe_a_remplacer';
```

Créer la base de données :

```sql
CREATE DATABASE hawkshot
    OWNER hawkshot_user;
```

Accorder les droits nécessaires :

```sql
GRANT ALL PRIVILEGES ON DATABASE hawkshot TO hawkshot_user;
```

Quitter :

```sql
\q
```

Tester la connexion :

```powershell
psql -U hawkshot_user -d hawkshot -h localhost
```

> PostgreSQL transforme par défaut les noms non entourés de guillemets en minuscules. Il est donc recommandé d'utiliser uniquement des noms en minuscules comme `hawkshot_user`.

---

## 12. Configurer les variables d'environnement

Ne jamais écrire les mots de passe directement dans le code ou dans un fichier suivi par Git.

À la racine du projet, créer un fichier `.env` à partir du modèle s'il existe :

```powershell
Copy-Item .env.example .env
```

Si aucun modèle n'existe encore, créer `.env` avec une configuration de ce type :

```dotenv
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=hawkshot
POSTGRES_USER=hawkshot_user
POSTGRES_PASSWORD=mot_de_passe_a_remplacer

DATABASE_URL=postgresql://hawkshot_user:mot_de_passe_a_remplacer@localhost:5432/hawkshot
```

Vérifier que `.env` est bien présent dans `.gitignore` :

```powershell
git check-ignore .env
```

La commande doit afficher :

```text
.env
```

Ne jamais exécuter :

```powershell
git add -f .env
```

---

## 13. Ajouter les données du projet

Les fichiers de données volumineux ne doivent généralement pas être versionnés dans Git.

Créer le dossier prévu par le projet, par exemple :

```powershell
mkdir data\raw
mkdir data\interim
mkdir data\processed
```

Déposer le jeu de données brut dans :

```text
data/raw/
```

Pour le dataset NASA C-MAPSS, la structure pourra par exemple ressembler à :

```text
data/
├── raw/
│   └── cmapss/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       ├── RUL_FD001.txt
│       ├── train_FD002.txt
│       ├── test_FD002.txt
│       ├── RUL_FD002.txt
│       ├── train_FD003.txt
│       ├── test_FD003.txt
│       ├── RUL_FD003.txt
│       ├── train_FD004.txt
│       ├── test_FD004.txt
│       └── RUL_FD004.txt
├── interim/
└── processed/
```

Vérifier que les données brutes sont ignorées par Git si elles ne doivent pas être publiées :

```powershell
git status
```

---

## 14. Vérification complète de l'installation

Depuis la racine de Hawkshot, exécuter successivement :

```powershell
git status
uv --version
uv run python --version
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pre-commit run --all-files
```

Vérifier également PostgreSQL :

```powershell
psql -U hawkshot_user -d hawkshot -h localhost
```

Puis dans `psql` :

```sql
SELECT current_database(), current_user;
```

Résultat attendu :

```text
current_database | current_user
-----------------+---------------
hawkshot         | hawkshot_user
```

Une fois toutes ces vérifications réussies, l'environnement de développement est prêt.

---

## 15. Commandes courantes

### Mettre le projet à jour

```powershell
git switch main
git pull origin main
uv sync
```

### Créer une branche

```powershell
git switch -c feature/nom-de-la-fonctionnalite
```

### Voir les modifications

```powershell
git status
git diff
```

### Vérifier le projet avant un commit

```powershell
uv run ruff check .
uv run ruff format .
uv run pytest
uv run pre-commit run --all-files
```

### Créer un commit

```powershell
git add .
git commit -m "type: description du changement"
```

Exemples de types :

```text
feat: nouvelle fonctionnalité
fix: correction d'un bug
test: ajout ou modification de tests
docs: documentation
refactor: restructuration sans changement fonctionnel
chore: configuration ou maintenance
```

### Envoyer la branche sur GitHub

```powershell
git push -u origin nom-de-la-branche
```

### Ajouter une dépendance

Dépendance nécessaire au fonctionnement du projet :

```powershell
uv add nom-du-paquet
```

Dépendance réservée au développement :

```powershell
uv add --dev nom-du-paquet
```

Après l'ajout, versionner les deux fichiers modifiés :

```text
pyproject.toml
uv.lock
```

---

## 16. Problèmes fréquents

### `uv` n'est pas reconnu

Fermer puis rouvrir PowerShell et VS Code.

Tester :

```powershell
uv --version
```

Si nécessaire, relancer l'installation de `uv`.

### PowerShell bloque l'activation de `.venv`

Utiliser directement les commandes `uv run`, ou autoriser les scripts pour l'utilisateur courant :

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### VS Code n'utilise pas le bon Python

Sélectionner manuellement :

```text
.venv\Scripts\python.exe
```

Puis vérifier dans le terminal intégré :

```powershell
python --version
```

ou, de préférence :

```powershell
uv run python --version
```

### `psql` n'est pas reconnu

Ajouter le dossier suivant au `PATH` Windows :

```text
C:\Program Files\PostgreSQL\<VERSION>\bin
```

Puis fermer et rouvrir le terminal.

### Échec d'authentification PostgreSQL

Vérifier :

- le nom d'utilisateur ;
- la casse du nom ;
- le mot de passe ;
- le port ;
- le nom de la base.

Tester explicitement :

```powershell
psql -h localhost -p 5432 -U hawkshot_user -d hawkshot
```

### Pre-commit modifie les fichiers pendant le commit

C'est un comportement normal.

Exécuter :

```powershell
git add .
uv run pre-commit run --all-files
git add .
git commit -m "type: description"
```

### Un dossier `__pycache__` est déjà suivi par Git

Le retirer de l'index sans le supprimer localement :

```powershell
git rm -r --cached .
git add .
git commit -m "chore: remove ignored files from Git tracking"
```

Vérifier que `.gitignore` contient notamment :

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.venv/
.env
```

---

## 17. Mise à jour de l'environnement

Après un `git pull`, si `pyproject.toml` ou `uv.lock` a changé :

```powershell
uv sync
```

Mettre à jour les environnements pre-commit si leur configuration a changé :

```powershell
uv run pre-commit install
uv run pre-commit run --all-files
```

Pour mettre à jour les dépendances du projet, cette opération doit être faite volontairement puis testée :

```powershell
uv lock --upgrade
uv sync
uv run pytest
uv run ruff check .
```

Ne pas mettre à jour toutes les dépendances sans vérifier que le projet fonctionne toujours.

---

## 18. Checklist finale

- [ ] VS Code est installé.
- [ ] Git est installé et configuré.
- [ ] Le dépôt Hawkshot est cloné.
- [ ] `uv` est installé.
- [ ] Python 3.11 est disponible.
- [ ] `uv sync` se termine sans erreur.
- [ ] VS Code utilise `.venv`.
- [ ] Ruff passe sans erreur.
- [ ] pytest passe sans erreur.
- [ ] pre-commit est installé.
- [ ] PostgreSQL est installé.
- [ ] La base `hawkshot` existe.
- [ ] L'utilisateur `hawkshot_user` peut se connecter.
- [ ] Le fichier `.env` est configuré et ignoré par Git.
- [ ] Les données brutes sont placées dans le bon dossier.
- [ ] `git status` ne montre aucun fichier sensible.
