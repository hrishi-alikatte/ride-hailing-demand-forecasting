\LoadClass[a4paper,12pt]{article} % Document class

%--------------------- Packages ------------------------

\RequirePackage[english]{babel} %Document language - CHANGED to English
\RequirePackage[utf8]{inputenc} %Special characters
\RequirePackage[section]{placeins}%Section placement
\RequirePackage[T1]{fontenc} %Font encoding
\RequirePackage{mathtools} %Math equations and symbols
\RequirePackage{siunitx} %Scientific notation (e.g.: \num{2e+9})
\RequirePackage{float} %Image placement
\RequirePackage{graphicx} %Image insertion
\RequirePackage[justification=centering]{caption} %Centered captions
\RequirePackage{subcaption}
\RequirePackage{wallpaper}
\RequirePackage{nomencl}
%\makenomenclature
\RequirePackage{fancyhdr}
%\pagestyle{fancy}
%\fancyheadoffset{1cm}
%\setlength{\headheight}{2cm}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[left=2.5cm,right=2.5cm,top=2cm,bottom=3.5cm]{geometry} %Page configuration


%-------------------- Report Information ----------------------

\newcommand{\UE}[1]{\renewcommand{\UE}{#1}}
\newcommand{\sujet}[1]{\renewcommand{\sujet}{#1}}
\newcommand{\titre}[1]{\renewcommand{\titre}{#1}}
\newcommand{\enseignant}[1]{\renewcommand{\enseignant}{#1}}
\newcommand{\eleves}[1]{\renewcommand{\eleves}{#1}}

\newcommand{\fairemarges}{
\makenomenclature
\pagestyle{fancy}
\fancyheadoffset{1cm}
\setlength{\headheight}{2cm}
\lhead{\includegraphics[scale=0.3]{logos/logo_ECL.jpg}}
\rhead{\nouppercase{\leftmark}}
\rfoot{\thepage}
\cfoot{\textbf{\titre}}
\lfoot{\UE}
}

\newcommand{\fairepagedegarde}{
\begin{titlepage}

\ThisLRCornerWallPaper{0.6}{logos/SteinECL.jpg}
	\centering
	\includegraphics[width=0.8\textwidth]{logos/logo.png}\par\vspace{1cm}
	{\scshape\LARGE \'Ecole Centrale Lyon \par}
	\vspace{1cm}
	{\scshape\Large \UE \\ \sujet \par}
	\vspace{0.5cm}
    \rule{\linewidth}{0.2 mm} \\[0.4 cm]
	{\huge\bfseries \titre \par} \
    \rule{\linewidth}{0.2 mm} \\[1 cm]
	\vspace{0.5cm}
    
	\begin{minipage}{0.5\textwidth}
		\begin{flushleft} \large
		\emph{\textbf{Students:}}\\
        \eleves\\
		\end{flushleft}
	\end{minipage}
	~
	\begin{minipage}{0.4\textwidth}
		\begin{flushright} \large
		\emph{\textbf{Supervisor:}} \\
		 \enseignant \\
		\end{flushright}
	\end{minipage}\\[2cm]
    
	\vfill
	{\large \today\par}

\end{titlepage}
}

\newcommand{\tabledematieres}{
\tableofcontents
\newpage
}

\newcommand{\insererfigure}[4]{
\begin{figure}[ht]
\centering
\includegraphics[height=#2]{#1}
\caption{#3}
\label{fig: #4}
\end{figure}
}
