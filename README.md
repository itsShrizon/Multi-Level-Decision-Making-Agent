# Multi-Level Chatbot API

A sophisticated FastAPI-based AI chatbot system for law firms with advanced message analysis, insights generation, and outbound messaging capabilities.

## Features

### 🤖 Chat Analysis
- **Message Triage**: Automatically classify messages as FLAG, IGNORE, or RESPOND
- **Risk Assessment**: Evaluate client retention risk with scoring
- **Sentiment Analysis**: Analyze client sentiment with numerical scoring
- **Event Detection**: Identify appointments and important dates
- **Response Generation**: Generate contextual responses based on analysis

### 📊 Insights Generation
- **Micro Insights**: Single-sentence client insights with sentiment embedding
- **High-Level Insights**: Comprehensive firm-wide analytics reports
- **Summary Insights**: Quick dashboard-ready insights

### 📧 Outbound Messaging
- **Proactive Messages**: Generate weekly check-ins and follow-ups
- **Appointment Reminders**: Automated appointment reminder scheduling
- **Case Updates**: Professional case progress communications
- **Message Scheduling**: Intelligent scheduling based on client preferences

### 🔧 Text Processing
- **Chat Summarization**: Generate conversation summaries
- **Text Concisification**: Make text concise while preserving meaning
- **Keyword Extraction**: Extract key terms from conversations
- **Urgency Classification**: Classify message urgency levels

## Architecture

```
apply-job-agent-backend/
├── app/
│   ├── core/               # Core application components
│   │   ├── config.py       # Configuration management
│   │   ├── dependencies.py # Dependency injection
│   │   ├── exceptions.py   # Custom exceptions
│   │   └── logging.py      # Logging configuration
│   ├── features/           # Feature modules
│   │   ├── chat/           # Chat analysis features
│   │   ├── insights/       # Insights generation
│   │   └── outbound/       # Outbound messaging
│   ├── shared/             # Shared components
│   │   ├── schemas.py      # Pydantic models
│   │   ├── utils.py        # Utility functions
│   │   └── middleware.py   # Custom middleware
│   └── main.py             # Application entry point
├── alembic/                # Database migrations
├── docker-compose.yml      # Docker configuration
└── requirements.txt        # Python dependencies
```

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- Redis (optional, for caching)
- PostgreSQL (optional, for data persistence)

### Installation

1. **Clone and setup the project:**
```bash
cd apply-job-agent-backend
cp .env.example .env
```

2. **Configure environment variables:**
Edit `.env` file with your settings:
```env
OPENAI_API_KEY=your-openai-api-key-here
ENVIRONMENT=development
DEBUG=true
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker

1. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

2. **Access the API:**
- API: http://localhost:8000
- Documentation: http://localhost:8000/api/docs
- Flower (Celery monitoring): http://localhost:5555

## API Endpoints

### Chat Analysis
- `POST /api/v1/chat/analyze` - Analyze client messages
- `POST /api/v1/chat/summarize` - Summarize conversations
- `POST /api/v1/chat/make-concise` - Make text concise

### Insights
- `POST /api/v1/insights/micro` - Generate micro insights
- `POST /api/v1/insights/high-level` - Generate firm insights
- `POST /api/v1/insights/summary` - Generate summary insights

### Outbound Messaging
- `POST /api/v1/outbound/generate` - Generate outbound messages
- `POST /api/v1/outbound/follow-up` - Generate follow-up messages
- `POST /api/v1/outbound/appointment-reminder` - Generate appointment reminders
- `POST /api/v1/outbound/case-update` - Generate case updates

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ENVIRONMENT` | Environment (development/production) | development |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `DATABASE_URL` | PostgreSQL connection URL | Optional |
| `REDIS_URL` | Redis connection URL | Optional |
| `RATE_LIMIT_REQUESTS` | Rate limit per minute | 100 |
| `MAX_CONVERSATION_HISTORY` | Max messages to process | 500 |

### OpenAI Models

- **GPT-4o**: Primary model for complex analysis
- **GPT-3.5-turbo**: For chat summarization
- **GPT-4o-mini**: For micro insights and quick tasks

## Features Deep Dive

### Message Analysis Pipeline

1. **Triage Agent**: Determines if message needs FLAG, IGNORE, or RESPOND
2. **Risk Agent**: Assesses client retention risk (Low/Medium/High + score)
3. **Sentiment Agent**: Analyzes sentiment (Positive/Neutral/Negative + score)
4. **Event Detection**: Identifies appointments and creates reminders
5. **Response Generator**: Creates contextual responses when needed

### Insights Engine

- **Micro Insights**: Real-time, single-sentence client insights
- **High-Level Reports**: Comprehensive business intelligence for firm leadership
- **Sentiment Tracking**: Continuous sentiment monitoring with historical context

### Outbound Messaging

- **Context-Aware**: Analyzes full conversation history for tone matching
- **Scheduling**: Intelligent message scheduling based on preferences
- **Multi-Type**: Weekly check-ins, appointment reminders, case updates

## Security Features

- **Rate Limiting**: Configurable request rate limiting
- **Input Validation**: Comprehensive input sanitization
- **Error Handling**: Structured error responses
- **CORS Configuration**: Configurable cross-origin policies
- **Trusted Hosts**: Host validation for production

## Monitoring & Logging

- **Structured Logging**: Comprehensive request/response logging
- **Health Checks**: Built-in health monitoring endpoints
- **Background Tasks**: Async analytics and logging
- **Error Tracking**: Detailed error logging with context

## Development

### Running Tests
```bash
pytest -v --cov=app tests/
```

### Code Formatting
```bash
black app/
isort app/
flake8 app/
```

### Type Checking
```bash
mypy app/
```

## Production Deployment

### Using Docker
```bash
docker-compose -f docker-compose.yml --profile production up -d
```

### Environment Setup
- Set `ENVIRONMENT=production`
- Use strong `SECRET_KEY`
- Configure proper database and Redis instances
- Set up SSL/TLS certificates
- Configure monitoring and logging

### Scaling Considerations
- Use Redis for distributed rate limiting
- Configure database connection pooling
- Set up load balancing for multiple instances
- Monitor OpenAI API usage and costs

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `/api/docs`
- ReDoc: `/api/redoc`
- OpenAPI JSON: `/api/openapi.json`

## Support

For issues and questions:
1. Check the logs: `docker-compose logs api`
2. Verify environment configuration
3. Check OpenAI API key and quotas
4. Review the API documentation

## License

This project is proprietary software. All rights reserved.
