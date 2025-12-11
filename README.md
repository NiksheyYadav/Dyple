# Dyple AI Platform

A unified AI Model Lifecycle Platform combining Training Engine, Playground, Workspace, and Developer Platform.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Docker & Docker Compose
- npm or yarn
- Azure Account (or Azurite for local development)
- Appwrite Account (for authentication)

### 1. Start Infrastructure

```bash
cd infra
docker-compose up -d
```

This starts:
- **PostgreSQL** (port 5432) - Database
- **Redis** (port 6379) - Job queue
- **Azurite** (ports 10000-10002) - Azure Storage Emulator

### 2. Configure Environment

```bash
cd backend
copy .env.example .env
```

Update `.env` with your Azure and Appwrite credentials:
- `AZURE_STORAGE_CONNECTION_STRING` - Azure Blob Storage connection string
- `APPWRITE_PROJECT_ID` - Appwrite project ID
- `APPWRITE_API_KEY` - Appwrite API key

For local development with Azurite:
```
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;
```

### 3. Start Backend

```bash
cd backend
npm install
npm run dev
```

API running at: http://localhost:3001
API Docs at: http://localhost:3001/api/docs

### 4. Start Frontend

```bash
npm install
npm run dev
```

Frontend running at: http://localhost:5173

## 📁 Project Structure

```
dyple/
├── src/                    # Frontend (React + Vite)
│   ├── components/ui/      # Reusable UI components
│   ├── pages/              # Page components
│   │   ├── Landing/        # Marketing landing page
│   │   ├── Auth/           # Sign in / Sign up pages
│   │   ├── Playground/     # AI chat & tools
│   │   ├── Workspace/      # Editor, files, tasks
│   │   ├── Training/       # ML training engine
│   │   └── Developer/      # API keys, analytics
│   ├── context/            # React context providers
│   └── theme/              # Design tokens
│
├── backend/                # Backend (NestJS)
│   └── src/
│       ├── entities/       # TypeORM entities
│       └── modules/        # Feature modules
│           ├── appwrite/   # Appwrite integration
│           ├── auth/       # JWT authentication
│           ├── storage/    # Azure Blob Storage
│           └── ...         # Other modules
│
└── infra/                  # Infrastructure
    └── docker-compose.yml  # Local dev stack
```

## 🎯 Features

### Training Engine
- **Datasets**: Upload, version, preview training data
- **Jobs**: Create and monitor training runs
- **Experiments**: Compare runs and metrics
- **Models**: Registry with version control
- **Deployments**: One-click model serving

### Playground
- **Chat**: AI chat interface with templates
- **Tools**: Summarize, paraphrase, image gen, etc.

### Workspace
- **Editor**: Markdown with AI assistance
- **Files**: Upload and manage documents
- **Tasks**: Kanban board for projects

### Developer Platform
- **API Keys**: Manage access credentials
- **Analytics**: Usage and cost tracking
- **API Playground**: Test endpoints interactively

## 🛠️ Tech Stack

**Frontend**
- React 19 + TypeScript
- Vite
- Tailwind CSS 4
- React Router

**Backend**
- NestJS
- TypeORM + PostgreSQL
- BullMQ + Redis
- Azure Blob Storage
- Appwrite (Authentication)
- Passport JWT

**Infrastructure**
- Docker Compose
- Azurite (Azure Storage Emulator)
- PostgreSQL
- Redis

## 📝 License

MIT
