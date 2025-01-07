@echo off
echo Starting deployment process...

REM Initialize git if not already initialized
if not exist .git (
    echo Initializing git repository...
    git init
) else (
    echo Git repository already initialized
)

REM Add all files to git
echo Adding files to git...
git add .

REM Commit changes
echo Committing changes...
git commit -m "Deployment update %date%"

REM Check if remote origin exists
git remote -v | findstr "origin" > nul
if errorlevel 1 (
    echo Adding remote origin...
    git remote add origin https://github.com/MuthuM3/pet-image-classifier.git
) else (
    echo Remote origin already exists
)

REM Push to main branch
echo Pushing to remote repository...
git push -u origin main

echo Deployment completed!
pause
