# HumanSQL - Natural Language to SQL Converter

Transform your questions into SQL queries effortlessly with HumanSQL, a modern Flask web application that converts natural language questions into SQL queries using Google's Gemini AI. Upload documents and query your data using plain English or voice commands.

## Features

- **Multi-Format Support**: CSV, Excel (.xlsx), PDF, and text files
- **AI-Powered**: Google Gemini AI for natural language processing
- **Voice Input**: Speak your queries with speech recognition
- **Professional UI**: Clean, responsive interface with collapsible sidebar
- **PostgreSQL Integration**: Automatic table creation and data import
- **Query History**: Track all queries and results with timestamps
- **Resume Processing**: Extract skills and features from resume documents
- **Export Results**: Download query results as CSV files
- **Keyboard Shortcuts**: Press Enter to execute queries
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

## Demo

1. **Upload** your data file (drag & drop supported)
2. **Ask** questions in natural language: "What are the top 5 sales?"
3. **Get** instant SQL queries and results
4. **Export** data as needed

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Google AI API key ([Get one here](https://makersuite.google.com/app/apikey))

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Balaswamyvasamsetti/File-to-SQL-Chatbot.git
cd File-to-SQL-Chatbot
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
# - Add your Google AI API key
# - Configure PostgreSQL settings
```

### 5. Set Up PostgreSQL
```sql
-- Create database (optional, uses 'postgres' by default)
CREATE DATABASE querycraft;
```

### 6. Run the Application
```bash
python app.py
```

Open your browser to `http://localhost:5000`

## Usage Examples

### Natural Language Queries
- "Show me all records where status is active"
- "What are the top 10 sales by amount?"
- "Count how many users registered this month"
- "Find all products with price greater than 100"

### Voice Commands
- Click the microphone button
- Speak your question clearly
- Query executes automatically

## Configuration

### Environment Variables (.env)
```env
# Google AI API Key
GOOGLE_AI_API_KEY=your_api_key_here

# PostgreSQL Configuration
DB_HOST=localhost
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432
```

### Supported File Types
- **CSV**: Comma-separated values
- **Excel**: .xlsx files (requires openpyxl)
- **PDF**: Text extraction with pdfplumber
- **Text**: Plain text or delimited data

## Architecture

```
querycraft/
├── app.py              # Main Flask application
├── static/
│   └── index.html      # Frontend interface
├── requirements.txt    # Python dependencies
├── .env.example       # Environment template
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Security

- Environment variables for sensitive data
- SQL injection protection with parameterized queries
- File type validation and sanitization
- CORS configuration for API security

## Technical Features

### Enhanced SQL Generation
- Smart query intent analysis
- Optimized query performance
- Automatic LIMIT clause handling
- Syntax validation and error prevention

### Responsive Design
- Mobile-first approach
- Touch-friendly interface
- Adaptive layouts for all screen sizes
- Collapsible sidebar for mobile devices

### Advanced File Processing
- Multi-format document support
- Intelligent text extraction
- Resume and document analysis
- Feature extraction capabilities

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Requirements

See `requirements.txt` for full dependency list. Key packages:
- Flask & Flask-CORS
- pandas & openpyxl
- psycopg2-binary
- pdfplumber & PyPDF2
- haystack-ai & google-ai-haystack

## Troubleshooting

### Common Issues

**Database Connection Error**
- Ensure PostgreSQL is running
- Check credentials in `.env` file
- Verify database exists

**API Key Error**
- Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Add to `.env` file
- Restart the application

**File Upload Issues**
- Check file format is supported
- Ensure file is not corrupted
- Try converting PDF to text-based format

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Google Gemini AI for natural language processing
- Flask community for the excellent framework
- Contributors and testers

---

Star this repository if you find it helpful!