#!/bin/bash
# Script para facilitar o uso do Docker com o sistema ComprasNet

echo "🐳 SISTEMA COMPRASNET - DOCKER MANAGER"
echo "======================================"

# Função para mostrar ajuda
show_help() {
    echo "Uso: $0 [comando]"
    echo ""
    echo "Comandos disponíveis:"
    echo "  build     - Construir a imagem Docker"
    echo "  run       - Executar o container"
    echo "  stop      - Parar o container"
    echo "  logs      - Ver logs do container"
    echo "  shell     - Acessar shell do container"
    echo "  clean     - Limpar containers e imagens"
    echo "  status    - Ver status dos containers"
    echo "  help      - Mostrar esta ajuda"
    echo ""
    echo "Exemplos:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 logs"
}

# Função para build
docker_build() {
    echo "🔨 Construindo imagem Docker..."
    docker build -t comprasnet-downloader .
    if [ $? -eq 0 ]; then
        echo "✅ Imagem construída com sucesso!"
    else
        echo "❌ Erro ao construir imagem"
        exit 1
    fi
}

# Função para executar
docker_run() {
    echo "🚀 Iniciando container ComprasNet..."
    
    # Verificar se já existe um container rodando
    if [ "$(docker ps -q -f name=comprasnet-auto)" ]; then
        echo "⚠️ Container já está rodando. Use 'docker logs -f comprasnet-auto' para ver logs."
        return
    fi
    
    # Executar com docker-compose se disponível
    if [ -f "docker-compose.yml" ]; then
        docker-compose up -d
    else
        # Executar com docker run simples
        docker run -d --name comprasnet-auto \
            -v $(pwd)/downloads_licitacoes:/app/downloads_licitacoes \
            -v $(pwd)/debug_sistema:/app/debug_sistema \
            comprasnet-downloader
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Container iniciado com sucesso!"
        echo "📝 Para ver logs: docker logs -f comprasnet-auto"
    else
        echo "❌ Erro ao iniciar container"
    fi
}

# Função para parar
docker_stop() {
    echo "🛑 Parando container..."
    if [ -f "docker-compose.yml" ]; then
        docker-compose down
    else
        docker stop comprasnet-auto
        docker rm comprasnet-auto
    fi
    echo "✅ Container parado!"
}

# Função para ver logs
docker_logs() {
    echo "📝 Logs do container:"
    docker logs -f comprasnet-auto
}

# Função para acessar shell
docker_shell() {
    echo "🐚 Acessando shell do container..."
    docker exec -it comprasnet-auto bash
}

# Função para limpeza
docker_clean() {
    echo "🧹 Limpando containers e imagens..."
    docker-compose down -v --rmi all 2>/dev/null || true
    docker stop comprasnet-auto 2>/dev/null || true
    docker rm comprasnet-auto 2>/dev/null || true
    docker rmi comprasnet-downloader 2>/dev/null || true
    echo "✅ Limpeza concluída!"
}

# Função para status
docker_status() {
    echo "📊 Status dos containers:"
    docker ps -a --filter name=comprasnet
    echo ""
    echo "📊 Imagens:"
    docker images | grep comprasnet
}

# Processar comando
case "$1" in
    build)
        docker_build
        ;;
    run)
        docker_run
        ;;
    stop)
        docker_stop
        ;;
    logs)
        docker_logs
        ;;
    shell)
        docker_shell
        ;;
    clean)
        docker_clean
        ;;
    status)
        docker_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Comando inválido: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
